#!/usr/bin/env python
# Copyright: Lars Andersen Bratholm - 2024

"""
Train a model.
"""

import os
import sys
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import contextlib

import numpy as np
import optuna
import pytorch_lightning as pl
import sklearn.model_selection
import tap
import torch
from loguru import logger
from numpy.typing import NDArray
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from torch import Tensor
from torch.utils.data import DataLoader

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append(DIR_PATH + "../")

from protein_classifier.trainer import LightningModel  # noqa:E402
from protein_classifier.utils import (  # noqa:E402
    configure_logger,
    load_yaml,
    load_pydantic_from_yaml,
)
from protein_classifier.data import Dataset  # noqa:E402
from protein_classifier.models import ModelParameters  # noqa:E402


class ArgumentParser(tap.Tap):
    """
    Train a model.
    """

    training_data: str
    test_data: Optional[str]
    model_parameters: str
    training_parameters: str
    output: str
    num_workers: int
    accelerator: Literal["gpu", "cpu"]

    def configure(self) -> None:
        self.add_argument(
            "training_data",
            nargs="?",
            help="Location of the training data csv file",
        )
        self.add_argument(
            "--test-data",
            "-t",
            nargs="?",
            default=None,
            help="Location of the test data csv file",
        )
        self.add_argument(
            "--model-parameters", "-m", nargs="?", help="Yaml file defining the Model"
        )
        self.add_argument(
            "--training-parameters",
            "-s",
            nargs="?",
            help="Yaml file defining the training settings",
        )
        self.add_argument("--output", "-o", nargs="?", help="The output folder.")
        self.add_argument(
            "--num-workers",
            "-w",
            nargs="?",
            help="The number of workers for the dataloader",
            default=0,
            type=int,
        )
        self.add_argument(
            "--accelerator",
            "-a",
            nargs="?",
            help="The accelerator ('gpu' or 'cpu')",
            default="gpu",
        )


def get_sequence_label_subset(
    sequences: List[str], labels: NDArray[np.int_], indices: NDArray[np.int_]
) -> Tuple[List[str], NDArray[np.int_]]:
    """
    Get a subset of the input

    :param sequences: the sequences
    :param labels: the labels
    :param indices: the subset indices
    :returns: sequence and label subsets
    """
    return ([sequences[i] for i in indices], labels[indices])


def create_dataloaders(
    args: ArgumentParser,
    batch_size: int,
) -> Tuple[
    DataLoader[Tuple[Tensor, Optional[Tensor]]],
    DataLoader[Tuple[Tensor, Optional[Tensor]]],
    DataLoader[Tuple[Tensor, Optional[Tensor]]],
]:
    """
    Create the data loaders

    :param args: command-line arguments
    :param batch_size: the batch_size
    :returns: training, validation and test dataloaders
    """
    logger.info("Creating dataloaders")
    if args.test_data is None:
        sequences = np.loadtxt(
            f"{args.training_data}", skiprows=1, usecols=0, delimiter=",", dtype=str
        ).tolist()
        labels = np.loadtxt(
            f"{args.training_data}", skiprows=1, usecols=1, delimiter=",", dtype=int
        )
        logger.info("Splitting training data into a 64%-16%-20% train/val/test split")
        indices = np.arange(labels.size)
        train_val_indices, test_indices = sklearn.model_selection.train_test_split(
            indices, test_size=0.2
        )
        train_indices, val_indices = sklearn.model_selection.train_test_split(
            train_val_indices, test_size=0.2
        )
        training_data = get_sequence_label_subset(sequences, labels, train_indices)
        validation_data = get_sequence_label_subset(sequences, labels, val_indices)
        testing_data = get_sequence_label_subset(sequences, labels, test_indices)
    else:
        train_sequences = np.loadtxt(
            f"{args.training_data}", skiprows=1, usecols=0, delimiter=",", dtype=str
        ).tolist()
        train_labels = np.loadtxt(
            f"{args.training_data}", skiprows=1, usecols=1, delimiter=",", dtype=int
        )
        logger.info("Splitting training data into a 80%-20% train/val split")
        indices = np.arange(train_labels.size)
        train_indices, val_indices = sklearn.model_selection.train_test_split(
            indices, test_size=0.2
        )
        training_data = get_sequence_label_subset(
            train_sequences, train_labels, train_indices
        )
        validation_data = get_sequence_label_subset(
            train_sequences, train_labels, val_indices
        )
        test_sequences: List[str] = np.loadtxt(
            f"{args.test_data}", skiprows=1, usecols=0, delimiter=",", dtype=str
        ).tolist()
        test_labels: NDArray[np.int_] = np.loadtxt(
            f"{args.test_data}", skiprows=1, usecols=1, delimiter=",", dtype=int
        )
        testing_data = (test_sequences, test_labels)

    num_workers = (
        len(os.sched_getaffinity(0)) if args.num_workers == 0 else args.num_workers
    )
    train_loader = DataLoader(
        Dataset(*training_data),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=Dataset.collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        Dataset(*validation_data),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=Dataset.collate,
    )
    test_loader = DataLoader(
        Dataset(*testing_data),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=Dataset.collate,
    )
    return train_loader, val_loader, test_loader


def check_loss(trainer: pl.Trainer) -> None:
    """
    If val_loss is nan or inf, raise exception.

    :param trainer: the pytorch lightning trainer
    """
    loss = trainer.logged_metrics["val_loss"]
    if torch.isnan(loss):
        logger.error("Stopped training as loss is nan")
        raise optuna.TrialPruned()
    if torch.isinf(loss):
        logger.error("Stopped training as loss is inf")
        raise optuna.TrialPruned()


def get_scheduler_optimizer_parameters(
    settings: Dict[str, Any],
    fused: bool,
) -> Tuple[Dict[str, Union[int, float]], Dict[str, Any]]:
    """
    Get scheduler and optimizer parameters from the settings.

    :param settings: the trainer settings
    :param fused: whether or not to use the fused implementation of the optimizer
    :returns: scheduler and optimizer parameters
    """
    scheduler_parameters = get_scheduler_parameters(settings)
    optimizer_parameters = {"lr": settings["lr"], "fused": fused}
    return scheduler_parameters, optimizer_parameters


def get_scheduler_parameters(
    settings: Dict[str, Any],
) -> Dict[str, Union[int, float]]:
    """
    Get scheduler parameters from the settings.

    :param settings: the trainer settings
    :returns: scheduler parameters
    """
    if "scheduler" not in settings or settings["scheduler"] is None:
        scheduler_parameters: Dict[str, Union[int, float]] = {}
    else:
        assert settings["scheduler"] == "cosine_annealing_lr"
        scheduler_parameters = {"T_max": settings["n_max_epochs"]}
    return scheduler_parameters


def train(  # pylint: disable=too-many-arguments
    output_folder: str,
    model: LightningModel,
    train_loader: DataLoader[Tuple[Tensor, Optional[Tensor]]],
    val_loader: DataLoader[Tuple[Tensor, Optional[Tensor]]],
    settings: Dict[str, Any],
    pre_training: bool = False,
    accelerator: Literal["gpu", "cpu"] = "gpu",
) -> LightningModel:
    """
    Training (or pre-training) of the model

    :param output_folder: the output folder
    :param model: the model
    :param train_loader: the training data loader
    :param val_loader: the validation data loader
    :param settings: the training settings
    :param pre_training: whether or not to do pre-training instead of
                         training on the target labels
    :returns: fitted model
    """
    if pre_training is True:
        if "pre_training" not in settings:
            return model
        basename = "pre_training"
        context = model.pre_training
    else:
        basename = "training"
        context = contextlib.nullcontext  # type: ignore[assignment]

    logger.info(f"Starting {basename}")
    training_settings = settings[f"{basename}"]

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=training_settings["early_stopping_patience"],
        divergence_threshold=0.1,
    )
    checkpoint = ModelCheckpoint(
        filename=f"{basename}_{{val_loss:.4g}}", monitor="val_loss", save_top_k=1
    )

    fused = (
        settings["gradient_clip_value"] == 0
        or settings["gradient_clip_algorithm"] is None
    )
    scheduler_parameters, optimizer_parameters = get_scheduler_optimizer_parameters(
        training_settings,
        fused,
    )

    model.set_optimizers(
        optimizer=training_settings["optimizer_name"],
        optimizer_parameters=optimizer_parameters,
        scheduler=training_settings["scheduler"],
        scheduler_parameters=scheduler_parameters,
    )
    trainer = pl.Trainer(
        max_epochs=training_settings["n_max_epochs"],
        accelerator=accelerator,
        precision=settings["precision"],
        devices=1,
        callbacks=[early_stopping, checkpoint],
        enable_progress_bar=False,
        gradient_clip_algorithm=settings["gradient_clip_algorithm"],
        gradient_clip_val=settings["gradient_clip_value"]
        if settings["gradient_clip_algorithm"] is not None
        else None,
        accumulate_grad_batches=settings["n_accumulate_grad"],
        default_root_dir=f"{output_folder}",
    )

    if (
        "freeze_embedding" in training_settings
        and training_settings["freeze_embedding"] is True
    ):
        with torch.no_grad():
            for p in model.model.embedding.parameters():
                p.requires_grad_(False)
    if (
        "freeze_encoder" in training_settings
        and training_settings["freeze_encoder"] is True
    ):
        with torch.no_grad():
            for p in model.model._encoder.parameters():
                p.requires_grad_(False)

    with context():
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

    check_loss(trainer)
    # Save checkpoint
    shutil.copy(checkpoint.best_model_path, f"{output_folder}/{basename}.ckpt")
    model = LightningModel.load_from_checkpoint(
        checkpoint_path=checkpoint.best_model_path
    )
    if "unfreeze" in training_settings and training_settings["unfreeze"] is True:
        with torch.no_grad():
            for p in model.model.embedding.parameters():
                p.requires_grad_(True)
            for p in model.model._encoder.parameters():
                p.requires_grad_(True)
    return model


def swa(  # pylint: disable=too-many-locals,too-many-arguments
    output_folder: str,
    model: LightningModel,
    train_loader: DataLoader[Tuple[Tensor, Optional[Tensor]]],
    val_loader: DataLoader[Tuple[Tensor, Optional[Tensor]]],
    settings: Dict[str, Any],
    accelerator: Literal["gpu", "cpu"] = "gpu",
) -> LightningModel:
    """
    Fine-tune the model bystochastic weight averaging

    :param output_folder: the output folder
    :param model: the model
    :param train_loader: the training data loader
    :param val_loader: the validation data loader
    :param settings: the training settings
    :returns: fitted model
    """
    if "swa" not in settings:
        return model

    logger.info("Starting stochastic weight averaging")
    checkpoint = ModelCheckpoint(
        filename="stage3_{val_loss:.4g}", monitor="val_loss", save_last=True
    )
    training_settings = settings["swa"]
    swa_callback = StochasticWeightAveraging(
        training_settings["lr"],
        swa_epoch_start=0.0,
        device=None,
        annealing_epochs=5,
    )
    # Only used to stop nan/inf
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=training_settings["n_epochs"] + 1,
        divergence_threshold=0.1,
    )

    optimizer_parameters = {"lr": training_settings["lr"], "fused": True}

    model.set_optimizers(
        optimizer=training_settings["optimizer_name"],
        optimizer_parameters=optimizer_parameters,
    )
    trainer = pl.Trainer(
        max_epochs=training_settings["n_epochs"],
        accelerator=accelerator,
        precision=settings["precision"],
        devices=1,
        callbacks=[checkpoint, early_stopping, swa_callback],
        enable_progress_bar=False,
        accumulate_grad_batches=settings["n_accumulate_grad"],
        default_root_dir=f"{output_folder}",
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    check_loss(trainer)

    shutil.copy(checkpoint.last_model_path, f"{output_folder}/swa_last.ckpt")

    assert swa_callback._average_model is not None
    model = swa_callback._average_model
    return model


def get_accuracy(
    model: LightningModel,
    dataloader: DataLoader[Tuple[Tensor, Optional[Tensor]]],
    output_folder: str,
    accelerator: Literal["gpu", "cpu"] = "gpu",
) -> float:
    """
    Make predictions on the test set and get the accuracy

    :param model: the model
    :param dataloader: the test dataloader
    :param output_folder: the output folder
    """
    trainer = pl.Trainer(
        accelerator=accelerator,
        enable_progress_bar=False,
        default_root_dir=f"{output_folder}",
    )
    predictions = torch.cat(trainer.predict(model, dataloader))  # type: ignore[arg-type]
    labels = torch.cat([batch[1] for batch in dataloader])
    accuracy = (predictions == labels).sum().item() / len(labels)
    with open(f"{output_folder}/test_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy:.4g}\n")
    with open(f"{output_folder}/test_predictions.txt", "w", encoding="utf-8") as f:
        f.write("prediction,target\n")
        for i in range(len(labels)):
            f.write(f"{predictions[i].item()},{labels[i].item()}\n")
    return accuracy


def main(args: ArgumentParser) -> float:
    """
    Train a model with the given settings.

    :param args: command-line arguments
    :returns: test loss
    """
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    configure_logger(sink=f"{args.output}/log.txt", log_level="DEBUG", mode="w+")

    settings = load_yaml(args.training_parameters)
    train_loader, val_loader, test_loader = create_dataloaders(
        args, settings["batch_size"]
    )
    model_parameters: ModelParameters = load_pydantic_from_yaml(  # type: ignore[assignment]
        ModelParameters,  # type: ignore[arg-type]
        args.model_parameters,
    )
    torch.set_float32_matmul_precision(settings["matmul_precision"])
    model = LightningModel(model_parameters)

    if "pre_training" in settings and isinstance(settings["pre_training"], str):
        model = LightningModel.load_from_checkpoint(settings["pre_training"])
    else:
        model = train(
            args.output,
            model,
            train_loader,
            val_loader,
            settings,
            pre_training=True,
            accelerator=args.accelerator,
        )
    model = train(
        args.output,
        model,
        train_loader,
        val_loader,
        settings,
        pre_training=False,
        accelerator=args.accelerator,
    )
    model = swa(args.output, model, train_loader, val_loader, settings)

    model.model.save_model(f"{args.output}/model.pt")
    test_accuracy = get_accuracy(
        model, test_loader, args.output, accelerator=args.accelerator
    )
    return test_accuracy


if __name__ == "__main__":
    parser = ArgumentParser()
    arguments = parser.parse_args()
    main(arguments)
