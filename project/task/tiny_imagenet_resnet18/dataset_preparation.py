"""Prepare and partition Tiny-ImageNet for federated learning."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from flwr.common.logger import log
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_and_extract_archive

HYDRA_FULL_ERROR = 1

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_FILENAME = "tiny-imagenet-200.zip"
EXTRACTED_DIRNAME = "tiny-imagenet-200"


def _download_data(dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    extracted_root = dataset_dir / EXTRACTED_DIRNAME

    if extracted_root.exists():
        log(logging.INFO, "Tiny-ImageNet already exists at %s", extracted_root)
        return extracted_root

    log(logging.INFO, "Downloading Tiny-ImageNet from %s", TINY_IMAGENET_URL)
    download_and_extract_archive(
        url=TINY_IMAGENET_URL,
        download_root=str(dataset_dir),
        filename=TINY_IMAGENET_FILENAME,
        remove_finished=False,
    )
    if not extracted_root.exists():
        raise FileNotFoundError(
            f"Expected extracted dataset at {extracted_root}, but it was not found."
        )
    return extracted_root


def _load_class_mapping(root: Path) -> tuple[list[str], dict[str, int]]:
    wnids_path = root / "wnids.txt"
    wnids = [line.strip() for line in wnids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
    return wnids, class_to_idx


def _collect_train_samples(root: Path, class_to_idx: dict[str, int]) -> tuple[list[str], list[int]]:
    train_dir = root / "train"
    paths: list[str] = []
    targets: list[int] = []

    for wnid, label in class_to_idx.items():
        image_dir = train_dir / wnid / "images"
        if not image_dir.exists():
            continue
        for img_path in sorted(image_dir.glob("*.JPEG")):
            paths.append(str(img_path.resolve()))
            targets.append(label)

    return paths, targets


def _collect_val_samples(root: Path, class_to_idx: dict[str, int]) -> tuple[list[str], list[int]]:
    val_dir = root / "val"
    image_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"

    filename_to_label: dict[str, int] = {}
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        filename, wnid = parts[0], parts[1]
        filename_to_label[filename] = class_to_idx[wnid]

    paths: list[str] = []
    targets: list[int] = []

    for img_path in sorted(image_dir.glob("*.JPEG")):
        paths.append(str(img_path.resolve()))
        targets.append(filename_to_label[img_path.name])

    return paths, targets


def _save_meta(path: Path, paths: list[str], targets: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"paths": paths, "targets": targets}, path)


def _iid_partition(
    targets: np.ndarray,
    num_clients: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(targets))
    rng.shuffle(indices)
    return [arr.astype(np.int64) for arr in np.array_split(indices, num_clients)]


def _dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    seed: int,
    min_size: int = 10,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    class_prob = rng.dirichlet([alpha] * num_clients, size=num_classes)

    while True:
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for cls in range(num_classes):
            cls_idx = np.where(targets == cls)[0]
            rng.shuffle(cls_idx)

            proportions = class_prob[cls]
            cut_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            splits = np.split(cls_idx, cut_points)

            for cid, split in enumerate(splits):
                client_indices[cid].extend(split.tolist())

        sizes = [len(v) for v in client_indices]
        if min(sizes) >= min_size:
            break

        class_prob = rng.dirichlet([alpha] * num_clients, size=num_classes)

    out = [np.array(v, dtype=np.int64) for v in client_indices]
    return out, class_prob


def _dirichlet_partition_with_given_probs(
    targets: np.ndarray,
    class_prob: np.ndarray,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    num_clients = class_prob.shape[1]
    num_classes = class_prob.shape[0]

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in range(num_classes):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)

        proportions = class_prob[cls]
        cut_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, cut_points)

        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())

    return [np.array(v, dtype=np.int64) for v in client_indices]


def _sort_indices_by_label(targets: np.ndarray) -> np.ndarray:
    return np.argsort(targets)


def _power_law_partition(
    targets: np.ndarray,
    num_clients: int,
    num_classes: int,
    seed: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)

    sorted_idx = _sort_indices_by_label(targets)
    sorted_targets = targets[sorted_idx]

    class_counts = np.bincount(sorted_targets, minlength=num_classes)
    labels_cs = np.cumsum(class_counts)
    labels_cs = np.concatenate([[0], labels_cs[:-1]])

    full_idx = np.arange(len(sorted_targets))
    hist = np.zeros(num_classes, dtype=np.int32)
    partitions_idx: list[list[int]] = [[] for _ in range(num_clients)]

    min_data_per_class = int(min_data_per_partition / num_labels_per_partition)

    for u_id in range(num_clients):
        for cls_offset in range(num_labels_per_partition):
            cls = (u_id + cls_offset) % num_classes
            start = labels_cs[cls] + hist[cls]
            stop = start + min_data_per_class
            chosen = full_idx[start:stop]
            partitions_idx[u_id].extend(sorted_idx[chosen].tolist())
            hist[cls] += len(chosen)

    probs = rng.lognormal(
        mean,
        sigma,
        size=(num_classes, max(1, int(np.ceil(num_clients / num_classes))), num_labels_per_partition),
    )
    remaining_per_class = class_counts - hist
    denom = np.sum(probs, axis=(1, 2), keepdims=True)
    probs = remaining_per_class.reshape(-1, 1, 1) * probs / np.maximum(denom, 1e-12)

    for u_id in range(num_clients):
        bucket = min(u_id // num_classes, probs.shape[1] - 1)
        for cls_offset in range(num_labels_per_partition):
            cls = (u_id + cls_offset) % num_classes
            count = int(probs[cls, bucket, cls_offset])
            start = labels_cs[cls] + hist[cls]
            stop = start + count
            chosen = full_idx[start:stop]
            partitions_idx[u_id].extend(sorted_idx[chosen].tolist())
            hist[cls] += len(chosen)

    return [np.array(v, dtype=np.int64) for v in partitions_idx]


def _shard_partition(
    targets: np.ndarray,
    num_clients: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    sorted_idx = np.argsort(targets)
    partition_size = len(targets) // num_clients
    shard_size = partition_size // 2
    num_shards = num_clients * 2

    shards = [
        sorted_idx[shard_size * idx : shard_size * (idx + 1)]
        for idx in range(num_shards)
    ]
    shard_perm = rng.permutation(num_shards)

    client_indices: list[np.ndarray] = []
    for i in range(num_clients):
        a = shards[shard_perm[2 * i]]
        b = shards[shard_perm[2 * i + 1]]
        client_indices.append(np.concatenate([a, b]).astype(np.int64))
    return client_indices


def _partition_train_and_val(
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    cfg: DictConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    num_clients = int(cfg.dataset.num_clients)
    num_classes = int(cfg.dataset.num_classes)
    seed = int(cfg.dataset.seed)

    if cfg.dataset.iid:
        train_parts = _iid_partition(train_targets, num_clients, seed)
        val_parts = _iid_partition(val_targets, num_clients, seed + 1)
        return train_parts, val_parts

    if cfg.dataset.power_law:
        train_parts = _power_law_partition(
            train_targets,
            num_clients=num_clients,
            num_classes=num_classes,
            seed=seed,
        )
        val_parts = _power_law_partition(
            val_targets,
            num_clients=num_clients,
            num_classes=num_classes,
            seed=seed + 1,
        )
        return train_parts, val_parts

    if cfg.dataset.lda:
        train_parts, class_prob = _dirichlet_partition(
            train_targets,
            num_clients=num_clients,
            num_classes=num_classes,
            alpha=float(cfg.dataset.lda_alpha),
            seed=seed,
        )
        val_parts = _dirichlet_partition_with_given_probs(
            val_targets,
            class_prob=class_prob,
            seed=seed + 1,
        )
        return train_parts, val_parts

    train_parts = _shard_partition(train_targets, num_clients, seed)
    val_parts = _shard_partition(val_targets, num_clients, seed + 1)
    return train_parts, val_parts


@hydra.main(
    config_path="../../conf",
    config_name="tiny_imagenet_resnet18",
    version_base=None,
)
def download_and_preprocess(cfg: DictConfig) -> None:
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    partition_dir = Path(cfg.dataset.partition_dir)
    if partition_dir.exists():
        log(logging.INFO, "Partitioning already exists at: %s", partition_dir)
        return

    extracted_root = _download_data(Path(cfg.dataset.dataset_dir))
    _, class_to_idx = _load_class_mapping(extracted_root)

    train_paths, train_targets = _collect_train_samples(extracted_root, class_to_idx)
    val_paths, val_targets = _collect_val_samples(extracted_root, class_to_idx)

    train_targets_np = np.array(train_targets, dtype=np.int64)
    val_targets_np = np.array(val_targets, dtype=np.int64)

    client_train_parts, client_val_parts = _partition_train_and_val(
        train_targets=train_targets_np,
        val_targets=val_targets_np,
        cfg=cfg,
    )

    partition_dir.mkdir(parents=True, exist_ok=True)

    _save_meta(partition_dir / "test_meta.pt", val_paths, val_targets)
    _save_meta(partition_dir / "train_meta.pt", train_paths, train_targets)

    for cid in range(int(cfg.dataset.num_clients)):
        client_dir = partition_dir / f"client_{cid}"
        client_dir.mkdir(parents=True, exist_ok=True)

        tr_idx = client_train_parts[cid]
        te_idx = client_val_parts[cid]

        client_train_paths = [train_paths[i] for i in tr_idx.tolist()]
        client_train_targets = train_targets_np[tr_idx].tolist()

        client_val_paths = [val_paths[i] for i in te_idx.tolist()]
        client_val_targets = val_targets_np[te_idx].tolist()

        _save_meta(client_dir / "train_meta.pt", client_train_paths, client_train_targets)
        _save_meta(client_dir / "test_meta.pt", client_val_paths, client_val_targets)

        log(
            logging.INFO,
            "client_%s -> train=%s, test=%s",
            cid,
            len(client_train_paths),
            len(client_val_paths),
        )


if __name__ == "__main__":
    download_and_preprocess()
