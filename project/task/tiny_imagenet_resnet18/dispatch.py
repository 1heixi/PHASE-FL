from pathlib import Path

from omegaconf import DictConfig

from project.task.default.dispatch import dispatch_config as dispatch_default_config
from project.task.tiny_imagenet_resnet18.dataset import get_dataloader_generators
from project.task.tiny_imagenet_resnet18.models import (
    get_tiny_network_generator_resnet_sparsyfed,
    get_tiny_network_generator_resnet_sparsyfed_no_act,
    get_tiny_network_generator_resnet_zerofl,
    get_tiny_resnet18,
)
from project.task.tiny_imagenet_resnet18.train_test import (
    get_fed_eval_fn,
    get_fixed_train_and_prune,
    get_train_and_prune,
    test_hetero_flash,
    train,
    test,
)
from project.types.common import DataStructure, TrainStructure


def dispatch_train(cfg: DictConfig) -> TrainStructure | None:
    train_structure: str | None = cfg.get("task", {}).get("train_structure", None)
    alpha = cfg.get("task", {}).get("alpha", 1.0)

    legacy_sparsity = cfg.get("task", {}).get("sparsity", None)
    run_cfg = cfg.get("task", {}).get("fit_config", {}).get("run_config", {})
    initial_sparsity = run_cfg.get("initial_sparsity", None)
    target_sparsity = run_cfg.get("target_sparsity", None)

    if target_sparsity is not None:
        effective_sparsity = float(target_sparsity)
    elif legacy_sparsity is not None:
        effective_sparsity = float(legacy_sparsity)
    else:
        effective_sparsity = 0.0

    mask = cfg.get("task", {}).get("mask", 0.0)
    effective_sparsity = effective_sparsity - mask

    if train_structure is not None and train_structure.upper() == "TINY_RN18":
        return (train, test, get_fed_eval_fn)

    if train_structure is not None and train_structure.upper() == "TINY_RN18_PRUNE":
        return (
            get_train_and_prune(alpha=alpha, amount=effective_sparsity, pruning_method="l1"),
            test,
            get_fed_eval_fn,
        )

    if train_structure is not None and train_structure.upper() == "TINY_RN18_FIX_PRUNE":
        return (
            get_fixed_train_and_prune(alpha=alpha, amount=effective_sparsity, pruning_method="l1"),
            test_hetero_flash,
            get_fed_eval_fn,
        )

    return None


def dispatch_data(cfg: DictConfig) -> DataStructure | None:
    client_model_and_data: str | None = cfg.get("task", {}).get("model_and_data", None)
    partition_dir: str | None = cfg.get("dataset", {}).get("partition_dir", None)

    if client_model_and_data is not None and partition_dir is not None:
        client_dataloader_gen, fed_dataloader_gen = get_dataloader_generators(Path(partition_dir))

        alpha: float = cfg.get("task", {}).get("alpha", 1.0)
        run_cfg = cfg.get("task", {}).get("fit_config", {}).get("run_config", {})
        initial_sparsity = run_cfg.get("initial_sparsity", None)
        legacy_sparsity = cfg.get("task", {}).get("sparsity", None)

        if initial_sparsity is not None:
            sparsity = float(initial_sparsity)
        elif legacy_sparsity is not None:
            sparsity = float(legacy_sparsity)
        else:
            sparsity = 0.0

        num_classes: int = cfg.get("dataset", {}).get("num_classes", 200)

        if client_model_and_data.upper() == "TINY_RN18":
            return (
                get_tiny_resnet18(num_classes=num_classes),
                client_dataloader_gen,
                fed_dataloader_gen,
            )

        if client_model_and_data.upper() == "TINY_SPARSYFED_RN18":
            return (
                get_tiny_network_generator_resnet_sparsyfed(
                    alpha=alpha, sparsity=sparsity, num_classes=num_classes
                ),
                client_dataloader_gen,
                fed_dataloader_gen,
            )

        if client_model_and_data.upper() == "TINY_SPARSYFED_NA_RN18":
            return (
                get_tiny_network_generator_resnet_sparsyfed_no_act(
                    alpha=alpha, sparsity=sparsity, num_classes=num_classes
                ),
                client_dataloader_gen,
                fed_dataloader_gen,
            )

        if client_model_and_data.upper() == "TINY_ZEROFL_RN18":
            return (
                get_tiny_network_generator_resnet_zerofl(
                    alpha=alpha, sparsity=sparsity, num_classes=num_classes
                ),
                client_dataloader_gen,
                fed_dataloader_gen,
            )

        if client_model_and_data.upper() == "TINY_FLASH_RN18":
            return (
                get_tiny_resnet18(num_classes=num_classes),
                client_dataloader_gen,
                fed_dataloader_gen,
            )

    return None


dispatch_config = dispatch_default_config
