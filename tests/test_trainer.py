# Copyright: Lars Andersen Bratholm - 2024

"""
Tests for models
"""

import contextlib
import os
import sys
from typing import Tuple, List
import pathlib

import numpy as np
from numpy.typing import NDArray
import pytest
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append(DIR_PATH + "../")


from protein_classifier.models import (  # noqa:E402
    ModelParameters,
    PreTrainingParameters,
)
from protein_classifier.trainer import (  # noqa:E402
    LightningModel,
)
from protein_classifier.data import (  # noqa:E402
    Dataset,
)


def _load_dataset() -> Tuple[List[str], NDArray[np.int_]]:
    sequences = np.loadtxt(
        f"{DIR_PATH}/data/dataset.csv", skiprows=1, usecols=0, delimiter=",", dtype=str
    ).tolist()
    labels = np.loadtxt(
        f"{DIR_PATH}/data/dataset.csv", skiprows=1, usecols=1, delimiter=",", dtype=int
    )
    return sequences, labels


@pytest.mark.parametrize("accelerator", ("cpu", "gpu"))
@pytest.mark.parametrize("pre_training", (True, False))
def test_training(tmp_path: pathlib.Path, accelerator: str, pre_training: bool) -> None:
    """
    Test that a the LightningModel class trains without errors
    """
    if (accelerator == "gpu") and (torch.cuda.is_available() is False):
        pytest.skip()

    batch_size = 2
    sequences, labels = _load_dataset()
    dataloader = DataLoader(
        Dataset(sequences, labels),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Dataset.collate,
    )

    # Change working dir
    os.chdir(tmp_path)

    parameters = ModelParameters(pre_training=PreTrainingParameters())
    model = LightningModel(parameters)

    model.set_optimizers(optimizer="Adam", optimizer_parameters={"lr": 0.01})
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="cpu",
        enable_progress_bar=False,
    )

    context = model.pre_training if pre_training is True else contextlib.nullcontext

    with context():  # type: ignore
        trainer.fit(model=model, train_dataloaders=dataloader)
