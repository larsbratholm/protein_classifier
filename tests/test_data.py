# Copyright: Lars Andersen Bratholm - 2024

"""
Tests for dataset
"""

import os
import sys

import pytest
import torch
from torch.utils.data import DataLoader

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append(DIR_PATH + "../")

from protein_classifier.data import (  # noqa:E402
    Dataset,
    load_dataset,
)
from protein_classifier.models import MAX_SEQUENCE_LENGTH  # noqa:E402


@pytest.mark.parametrize("batch_size", (1, 2))
def test_dataloader(batch_size: int) -> None:
    """
    Test that a Dataset class works as intended with the collate function that
    adds 0-padding.

    :param batch_size: the batch size
    """
    sequences, v_genes, j_genes, labels = load_dataset(f"{DIR_PATH}/data/dataset.csv")
    dataloader = DataLoader(
        Dataset(sequences, v_genes, j_genes, labels),
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=Dataset.collate,
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
    sequences, v_genes, j_genes, labels = load_dataset(f"{DIR_PATH}/data/dataset.csv")
    dataloader = DataLoader(
        Dataset(sequences, v_genes, j_genes, labels),
        batch_size=batch_size,
        shuffle=False,
        # collate_fn=Dataset.collate,
    )
    batch_sequences, batch_v_genes, batch_j_genes, batch_labels = next(iter(dataloader))

    expected_padded_size = MAX_SEQUENCE_LENGTH

    assert batch_sequences.shape == torch.Size((batch_size, expected_padded_size))
    assert batch_sequences.dtype == torch.int64

    for arr in (batch_labels, batch_v_genes, batch_j_genes):
        assert arr.shape == torch.Size((batch_size,))
        assert arr.dtype == torch.int64


def test_padding() -> None:
    """
    Test that 0-padding is added correctly
    """
    sequences, v_genes, j_genes, labels = load_dataset(f"{DIR_PATH}/data/dataset.csv")
    dataloader = DataLoader(
        Dataset(sequences, v_genes, j_genes, labels),
        batch_size=len(sequences),
        shuffle=False,
        # collate_fn=Dataset.collate,
    )
    batch_sequences, _, _, _ = next(iter(dataloader))
    assert (batch_sequences == 0).any()

    sequence_lengths = [len(item) + 2 for item in sequences]

    for item, size in zip(batch_sequences, sequence_lengths):
        assert (item[:size] != 0).all()
        assert (item[size:] == 0).all()
