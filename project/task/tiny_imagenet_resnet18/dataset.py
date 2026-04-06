"""Tiny-ImageNet dataset utilities for federated learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from project.task.default.dataset import (
    ClientDataloaderConfig as DefaultClientDataloaderConfig,
)
from project.task.default.dataset import (
    FedDataloaderConfig as DefaultFedDataloaderConfig,
)
from project.types.common import ClientDataloaderGen, FedDataloaderGen

ClientDataloaderConfig = DefaultClientDataloaderConfig
FedDataloaderConfig = DefaultFedDataloaderConfig


def _build_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _build_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        targets: list[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.paths = paths
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.paths[idx]).convert("RGB")
        target = int(self.targets[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def _load_meta(meta_path: Path) -> tuple[list[str], list[int]]:
    meta: dict[str, Any] = torch.load(meta_path)
    return list(meta["paths"]), list(meta["targets"])


def get_dataloader_generators(
    partition_dir: Path,
) -> tuple[ClientDataloaderGen, FedDataloaderGen]:
    def get_client_dataloader(
        cid: str | int,
        test: bool,
        _config: dict,
    ) -> DataLoader:
        config: ClientDataloaderConfig = ClientDataloaderConfig(**_config)
        client_dir = partition_dir / f"client_{cid}"

        if not test:
            paths, targets = _load_meta(client_dir / "train_meta.pt")
            transform = _build_train_transform()
        else:
            paths, targets = _load_meta(client_dir / "test_meta.pt")
            transform = _build_test_transform()

        dataset = TinyImageNetDataset(paths=paths, targets=targets, transform=transform)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=not test,
            num_workers=4,
            pin_memory=True,
        )

    def get_federated_dataloader(
        test: bool,
        _config: dict,
    ) -> DataLoader:
        config: FedDataloaderConfig = FedDataloaderConfig(**_config)

        if test:
            paths, targets = _load_meta(partition_dir / "test_meta.pt")
            transform = _build_test_transform()
        else:
            paths, targets = _load_meta(partition_dir / "train_meta.pt")
            transform = _build_train_transform()

        dataset = TinyImageNetDataset(paths=paths, targets=targets, transform=transform)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=not test,
            num_workers=4,
            pin_memory=True,
        )

    return get_client_dataloader, get_federated_dataloader
