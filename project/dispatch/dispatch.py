"""Dispatches the functionality of the task."""

from collections.abc import Callable

from omegaconf import DictConfig

from project.task.default.dispatch import dispatch_config as dispatch_default_config

from project.task.cifar_resnet18.dispatch import (
    dispatch_config as dispatch_resnet18_config,
)
from project.task.cifar_resnet18.dispatch import dispatch_data as dispatch_resnet18_data
from project.task.cifar_resnet18.dispatch import (
    dispatch_train as dispatch_resnet18_train,
)

from project.task.tiny_imagenet_resnet18.dispatch import (
    dispatch_config as dispatch_tiny_resnet18_config,
)
from project.task.tiny_imagenet_resnet18.dispatch import (
    dispatch_data as dispatch_tiny_resnet18_data,
)
from project.task.tiny_imagenet_resnet18.dispatch import (
    dispatch_train as dispatch_tiny_resnet18_train,
)

from project.task.speech_resnet18.dispatch import (
    dispatch_config as dispatch_speech_resnet18_config,
)
from project.task.speech_resnet18.dispatch import (
    dispatch_data as dispatch_speech_resnet18_data,
)
from project.task.speech_resnet18.dispatch import (
    dispatch_train as dispatch_speech_resnet18_train,
)

from project.task.cub_vit.dispatch import (
    dispatch_config as dispatch_vit_config,
)
from project.task.cub_vit.dispatch import dispatch_data as dispatch_vit_data
from project.task.cub_vit.dispatch import dispatch_train as dispatch_vit_train

from project.types.common import ConfigStructure, DataStructure, TrainStructure


def dispatch_train(cfg: DictConfig) -> TrainStructure:
    task_train_functions: list[Callable[[DictConfig], TrainStructure | None]] = [
        dispatch_resnet18_train,
        dispatch_tiny_resnet18_train,
        dispatch_speech_resnet18_train,
        dispatch_vit_train,
    ]

    for task in task_train_functions:
        result = task(cfg)
        if result is not None:
            return result

    raise ValueError(f"Unable to match the train/test and fed_test functions: {cfg}")


def dispatch_data(cfg: DictConfig) -> DataStructure:
    task_data_dependent_functions: list[Callable[[DictConfig], DataStructure | None]] = [
        dispatch_resnet18_data,
        dispatch_tiny_resnet18_data,
        dispatch_speech_resnet18_data,
        dispatch_vit_data,
    ]

    for task in task_data_dependent_functions:
        result = task(cfg)
        if result is not None:
            return result

    raise ValueError(
        f"Unable to match the net generator and dataloader generator functions: {cfg}"
    )


def dispatch_config(cfg: DictConfig) -> ConfigStructure:
    task_config_functions: list[Callable[[DictConfig], ConfigStructure | None]] = [
        dispatch_default_config,
        dispatch_resnet18_config,
        dispatch_tiny_resnet18_config,
        dispatch_speech_resnet18_config,
        dispatch_vit_config,
    ]

    for task in task_config_functions:
        result = task(cfg)
        if result is not None:
            return result

    raise ValueError(f"Unable to match the config generation functions: {cfg}")
