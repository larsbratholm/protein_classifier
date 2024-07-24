# Copyright Lars Andersen Bratholm - 2024

"""
Training class.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from loguru import logger
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from .models import Model, ModelParameters
from .utils import configure_logger, name_to_optimizer, name_to_scheduler

# Set default logging
configure_logger(sink=None, log_level="INFO")


class LightningModel(pl.LightningModule):
    """
    PyTorch Lightning wrapper for training models.
    """

    def __init__(self, parameters: ModelParameters = ModelParameters()) -> None:
        """
        :param parameters: parameters defining the model (see ModelParameters docs)
        """
        super().__init__()

        # Used in checkpointing
        self.save_hyperparameters("parameters")

        model = Model(parameters)
        self.model = torch.compile(model)
        loss = nn.CrossEntropyLoss()
        self.loss = torch.compile(loss)

        # Set optimizer and scheduler defaults
        self._optimizer: str = "adam"
        self._scheduler: Optional[str] = None
        self._optimizer_parameters: Dict[str, Any] = {}
        self._scheduler_parameters: Dict[str, Any] = {}
        self.training_step_outputs: List[Dict[str, Union[float, int]]] = []
        self.validation_step_outputs: List[Dict[str, Union[float, int]]] = []
        self.validation_losses: List[float] = []

    def set_optimizers(
        self,
        optimizer: str = "adam",
        scheduler: Optional[str] = None,
        optimizer_parameters: Optional[Dict[str, Any]] = None,
        scheduler_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set the optimizer and scheduler.

        :param optimizer: name of the optimizer
        :param scheduler: name of the scheduler
        :param optimizer_parameters: parameters for the optimizer
        :param scheduler_parameters: parameters for the scheduler
        """
        self._optimizer = optimizer
        self._scheduler = scheduler
        if optimizer_parameters is None:
            optimizer_parameters = {}
        if scheduler_parameters is None:
            scheduler_parameters = {}
        self._optimizer_parameters = optimizer_parameters
        self._scheduler_parameters = scheduler_parameters

    def configure_optimizers(self) -> tuple[list[Optimizer], list[Dict[str, Any]]]:  # type: ignore[override]
        """
        Configure optimizer and scheduler for training.

        :returns: the initialized optimizer and scheduler.
        """
        # Setup model optimizer and scheduler
        optimizer_parameters_ = copy.deepcopy(self._optimizer_parameters)
        optimizer_parameters_["params"] = list(self.parameters())
        optimizer = name_to_optimizer(self._optimizer, optimizer_parameters_)
        scheduler_parameters_ = copy.deepcopy(self._scheduler_parameters)
        scheduler_parameters_["optimizer"] = optimizer
        scheduler_configs = {
            "scheduler": name_to_scheduler(self._scheduler, scheduler_parameters_),
            "interval": "step" if self._scheduler == "onecycle_lr" else "epoch",
        }
        if isinstance(scheduler_configs["scheduler"], ReduceLROnPlateau):
            scheduler_configs["monitor"] = "val_loss"

        return [optimizer], [scheduler_configs]

    def forward(  # pylint: disable=arguments-differ,unused-argument
        self,
        batch: tuple[Tensor, Tensor, Tensor, list[None] | Tensor],
        batch_idx: int = 0,
    ) -> tuple[Tensor, Tensor | List[None]]:
        """
        Do predictions on the batch

        :param batch: features and targets
        :param batch_idx: index of the current batch
        :returns: The logits and labels
        """
        sequences, v_genes, j_genes, labels = batch

        logits: Tensor = self.model(sequences, v_genes, j_genes)
        return logits, labels

    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, list[None] | Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """
        Train the model for one batch.

        :param batch: features and targets
        :param batch_idx: index of the current batch
        :param dataloader_idx: index of the current dataloader
        :returns: The training loss
        """
        logits, labels = self.forward(batch)
        loss: Tensor = self.loss(logits, labels)

        self.training_step_outputs.append(
            {"loss": loss.item(), "batch_size": logits.shape[0]}
        )
        return loss

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, list[None] | Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """
        Validate the model on a batch.

        :param batch: features and targets
        :param batch_idx: index of the current batch
        :param dataloader_idx: index of the current dataloader
        :returns: validation loss
        """
        logits, labels = self.forward(batch)
        loss: Tensor = self.loss(logits, labels)

        self.validation_step_outputs.append(
            {"loss": loss.item(), "batch_size": logits.shape[0]}
        )
        return loss

    def on_train_epoch_end(self) -> None:
        """
        Create summary statistics of the training data for the epoch.
        """
        total_loss = torch.sum(
            torch.tensor([x["loss"] for x in self.training_step_outputs])
        )
        total_size = torch.sum(
            torch.tensor([x["batch_size"] for x in self.training_step_outputs])
        )
        train_loss = total_loss / total_size
        self.log("train_loss", train_loss, on_epoch=True)
        logger.info(f"Epoch: {self.current_epoch} | Training loss: {train_loss:.3g}")
        self.training_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self) -> None:
        """
        Create summary statistics of the validation data for the epoch.
        """
        total_loss = torch.sum(
            torch.tensor([x["loss"] for x in self.validation_step_outputs])
        )
        total_size = torch.sum(
            torch.tensor([x["batch_size"] for x in self.validation_step_outputs])
        )
        val_loss = (total_loss / total_size).item()
        self.validation_losses.append(val_loss)
        self.log("val_loss", val_loss, on_epoch=True)
        logger.info(f"Epoch: {self.current_epoch} | Validation loss: {val_loss:.3g}")
        self.validation_step_outputs.clear()  # Free memory
        best_val_loss = min(self.validation_losses)
        self.log("best_val_loss", best_val_loss, on_epoch=True)

    def predict_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, list[None] | Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        """
        Sample labels from the logits at 0 temperature.

        :param batch: features and targets
        :param batch_idx: index of the current batch
        :param dataloader_idx: index of the current dataloader
        :returns: sampled labels
        """
        logits, _ = self.forward(batch)
        temperature = 1e-6
        p = F.softmax(logits / temperature, dim=1)
        sampled_labels = torch.multinomial(p, 1)[:, 0]
        return sampled_labels
