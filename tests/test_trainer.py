# Copyright: Lars Andersen Bratholm - 2024

"""
Tests for models
"""

import os
import sys
import pathlib

import pytest
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append(DIR_PATH + "../")


from protein_classifier.models import (  # noqa:E402
    ModelParameters,
)
from protein_classifier.trainer import (  # noqa:E402
    LightningModel,
)
from protein_classifier.data import (  # noqa:E402
    Dataset,
    load_dataset,
)


@pytest.mark.parametrize("accelerator", ("cpu", "gpu"))
def test_training(tmp_path: pathlib.Path, accelerator: str) -> None:
    """
    Test that a the LightningModel class trains without errors
    """
    if (accelerator == "gpu") and (torch.cuda.is_available() is False):
        pytest.skip()

    batch_size = 2
    sequences, v_genes, j_genes, labels = load_dataset(f"{DIR_PATH}/data/dataset.csv")
    dataloader = DataLoader(
        Dataset(sequences, v_genes, j_genes, labels),
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=Dataset.collate,
    )

    # Change working dir
    os.chdir(tmp_path)

    parameters = ModelParameters()
    model = LightningModel(parameters)

    model.set_optimizers(optimizer="Adam", optimizer_parameters={"lr": 0.01})
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="cpu",
        enable_progress_bar=False,
    )

    trainer.fit(model=model, train_dataloaders=dataloader)
