# Copyright: Lars Andersen Bratholm - 2024

"""
Tests for dataset
"""

import os
import sys
from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray
import pytest
import torch
from torch.utils.data import DataLoader

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append(DIR_PATH + "../")

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


@pytest.mark.parametrize("batch_size", (1, 2))
def test_dataloader(batch_size: int) -> None:
    """
    Test that a Dataset class works as intended with the collate function that
    adds 0-padding.

    :param batch_size: the batch size
    """
    sequences, labels = _load_dataset()
    dataloader = DataLoader(
        Dataset(sequences, labels),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Dataset.collate,
    )
    for _ in range(3):
        for _ in dataloader:
            pass


@pytest.mark.parametrize("batch_size", (1, 2))
def test_Dataset(batch_size: int) -> None:
    """
    Test that Dataset returns arrays of correct shape and type.

    :param batch_size: the batch_size
    """
    sequences, labels = _load_dataset()
    dataloader = DataLoader(
        Dataset(sequences, labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=Dataset.collate,
    )
    batch_sequences, batch_labels = next(iter(dataloader))

    expected_padded_size = max(len(item) for item in sequences[:batch_size]) + 2
    assert batch_sequences.shape == torch.Size((batch_size, expected_padded_size))
    assert batch_labels.shape == torch.Size((batch_size,))

    assert batch_sequences.dtype == torch.int64
    assert batch_labels.dtype == torch.int64


def test_padding() -> None:
    """
    Test that 0-padding is added correctly
    """
    sequences, labels = _load_dataset()
    dataloader = DataLoader(
        Dataset(sequences, labels),
        batch_size=len(sequences),
        shuffle=False,
        collate_fn=Dataset.collate,
    )
    batch_sequences, _ = next(iter(dataloader))
    assert (batch_sequences == 0).any()

    sequence_lengths = [len(item) + 2 for item in sequences]

    for item, size in zip(batch_sequences, sequence_lengths):
        assert (item[:size] != 0).all()
        assert (item[size:] == 0).all()
