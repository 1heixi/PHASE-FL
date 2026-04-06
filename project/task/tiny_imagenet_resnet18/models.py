"""Tiny-ImageNet models."""

from __future__ import annotations

from collections.abc import Callable

from project.task.cifar_resnet18.models import (
    NetCifarResnet18,
    get_network_generator_resnet_sparsyfed,
    get_network_generator_resnet_sparsyfed_no_act,
    get_network_generator_resnet_zerofl,
    get_parameters_to_prune,
    get_resnet18,
    init_weights,
    replace_layer_with_sparsyfed,
    replace_layer_with_sparsyfed_no_act,
    replace_layer_with_swat,
    set_spectral_global_exponent,
)

NetTinyImageNetResnet18 = NetCifarResnet18


def get_tiny_resnet18(num_classes: int = 200) -> Callable[[dict], NetTinyImageNetResnet18]:
    return get_resnet18(num_classes=num_classes)


def get_tiny_network_generator_resnet_sparsyfed(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 200,
    pruning_type: str = "unstructured",
):
    return get_network_generator_resnet_sparsyfed(
        alpha=alpha,
        sparsity=sparsity,
        num_classes=num_classes,
        pruning_type=pruning_type,
    )


def get_tiny_network_generator_resnet_sparsyfed_no_act(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 200,
):
    return get_network_generator_resnet_sparsyfed_no_act(
        alpha=alpha,
        sparsity=sparsity,
        num_classes=num_classes,
    )


def get_tiny_network_generator_resnet_zerofl(
    alpha: float = 1.0,
    sparsity: float = 0.0,
    num_classes: int = 200,
    pruning_type: str = "unstructured",
):
    return get_network_generator_resnet_zerofl(
        alpha=alpha,
        sparsity=sparsity,
        num_classes=num_classes,
        pruning_type=pruning_type,
    )


__all__ = [
    "NetTinyImageNetResnet18",
    "get_tiny_resnet18",
    "get_tiny_network_generator_resnet_sparsyfed",
    "get_tiny_network_generator_resnet_sparsyfed_no_act",
    "get_tiny_network_generator_resnet_zerofl",
    "get_parameters_to_prune",
    "replace_layer_with_swat",
    "replace_layer_with_sparsyfed",
    "replace_layer_with_sparsyfed_no_act",
    "init_weights",
    "set_spectral_global_exponent",
]
