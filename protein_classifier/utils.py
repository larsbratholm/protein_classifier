# Copyright Lars Andersen Bratholm - 2024

"""
Utilities.
"""

import functools
import io
import logging
import sys
import json
import warnings
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, conint, confloat
import torch
import yaml
from loguru import logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)


_NAME_TO_ACTIVATION: Dict[str, Any] = {
    "elu": torch.nn.ELU,  # Special case of CELU
    "hardshrink": torch.nn.Hardshrink,
    "hardsigmoid": torch.nn.Hardsigmoid,
    "hardtanh": torch.nn.Hardtanh,
    "hardswish": torch.nn.Hardswish,
    "leakyrelu": torch.nn.LeakyReLU,
    "logsigmoid": torch.nn.LogSigmoid,
    "prelu": torch.nn.PReLU,
    "relu": torch.nn.ReLU,
    "relu6": torch.nn.ReLU6,
    "rrelu": torch.nn.RReLU,
    "selu": torch.nn.SELU,
    "celu": torch.nn.CELU,
    "gelu": torch.nn.GELU,
    "sigmoid": torch.nn.Sigmoid,
    "silu": torch.nn.SiLU,
    "mish": torch.nn.Mish,
    "softplus": torch.nn.Softplus,
    "softshrink": torch.nn.Softshrink,
    "softsign": torch.nn.Softsign,
    "tanh": torch.nn.Tanh,
    "tanhshrink": torch.nn.Tanhshrink,
}


_NAME_TO_OPTIMIZER: Dict[str, Any] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}

_NAME_TO_SCHEDULER: Dict[str, Any] = {
    "cosine_annealing_lr": torch.optim.lr_scheduler.CosineAnnealingLR,
    "onecycle_lr": torch.optim.lr_scheduler.OneCycleLR,
}


# Types
NaturalNumber = conint(gt=0)
NaturalNumberMultipleOfTwo = conint(gt=0, multiple_of=2)
WholeNumber = conint(ge=0)
Percentile = confloat(ge=0, le=1)


def load_yaml(yaml_file: str) -> Dict[str, Any]:
    """
    Load a yaml file.

    :param yaml_file: the yaml filename
    :returns: the yaml file as dictionary
    """
    with open(yaml_file, encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    return data


def save_yaml(data: Dict[str, Any], output: str) -> None:
    """
    Save a yaml file.

    :param data: the data to save
    :param output: the filename of the yaml output
    """
    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)


def name_to_activation(name: str, parameters: Dict[str, Any]) -> torch.nn.Module:
    """
    Get an activation function by its name.

    :param name: name of activation function
    :param parameters: parameters to the activation function
    :returns: the activation function module
    """

    name = name.lower()

    if name not in _NAME_TO_ACTIVATION:
        raise ValueError(f"{name} not recognized as a valid ")

    activation_function = _NAME_TO_ACTIVATION[name]
    initialized_activation_function: torch.nn.Module = activation_function(**parameters)
    return initialized_activation_function


def name_to_optimizer(name: str, parameters: Dict[str, Any]) -> Optimizer:
    """
    Get an optimizer function by its name.

    :param name: name of the optimization function
    :param parameters: parameters of the optimization function
    :returns: the optimizer
    """

    name = name.lower()

    if name not in _NAME_TO_OPTIMIZER:
        raise KeyError(f"{name} not recognized as a valid optimizer.")

    optimizer = _NAME_TO_OPTIMIZER[name]
    initialized_optimizer: Optimizer = optimizer(**parameters)
    return initialized_optimizer


def name_to_scheduler(name: Optional[str], parameters: Dict[str, Any]) -> _LRScheduler:
    """
    Get a scheduler function by its name. Uses ConstantLR as a dummy
    scheduler in case name is None.

    :param name: name of the scheduler class
    :param parameters: parameters of the scheduler class
    :returns: the scheduler
    """

    if name is None:
        scheduler: Any = torch.optim.lr_scheduler.ConstantLR
        parameters["factor"] = 1
        parameters["total_iters"] = 0
    else:
        name = name.lower()
        if name not in _NAME_TO_SCHEDULER:
            raise KeyError(f"{name} not recognized as a valid scheduler.")

        scheduler = _NAME_TO_SCHEDULER[name]

    initialized_scheduler: _LRScheduler = scheduler(**parameters)
    return initialized_scheduler


def configure_logger(
    sink: Union[str, io.TextIOBase, None, logging.StreamHandler] = None,  # type: ignore[type-arg]
    log_level: Union[str, int] = "INFO",
    mode: str = "a",
) -> None:
    """
    Configure logger.

    :param sink: the file location/stream/handler for the log. If None defaults to stderr
    :param log_level: the logging level
    :param mode: the write mode
    """
    if sink is None:
        sink = sys.stderr  # type: ignore[assignment]
    if isinstance(sink, str):
        add_fn = functools.partial(logger.add, mode=mode)
    else:
        add_fn = logger.add  # type: ignore[assignment]

    logger.remove()
    add_fn(
        sink,
        level=log_level,
        format="<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <level>{level}</> "
        "| <c>{module}</>:<c>{line}</> | <level>{message}</>",
    )


# NOTE: should type below three functions as overload for each pydantic object
def save_pydantic_as_yaml(obj: BaseModel, filename: str) -> None:
    """
    Save a pydantic object as yaml

    :param obj: the pydantic object
    :param filename: the filename to save to
    """
    with open(filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj.dict(), f)


def load_pydantic_from_yaml(parent: BaseModel, filename: str) -> BaseModel:
    """
    Load pydantic object from filename

    :param parent: the parent object
    :param filename: the filename to load from
    :returns: initialized parent object
    """
    with open(filename, "r", encoding="utf-8") as f:
        parameters = yaml.safe_load(f)
    return dict_to_pydantic(parent, parameters)


def dict_to_pydantic(parent: BaseModel, parameters: Dict[str, Any]) -> BaseModel:
    """
    Convert dictionary to pydantic object

    :param parent: the parent object
    :param parameters: the dictionary
    :returns: initialized parent object
    """
    obj = parent.parse_raw(json.dumps(parameters))
    return obj
