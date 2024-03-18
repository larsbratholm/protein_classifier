# Copyright: Lars Andersen Bratholm - 2024

"""
Tests for models
"""

import os
import sys
from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append(DIR_PATH + "../")


from protein_classifier.models import (  # noqa:E402
    Model,
    _DenseLayer,
    _SkipGram,
    _SkipGramParameters,
    PreTrainingParameters,
    ModelParameters,
)
from protein_classifier.data import (  # noqa:E402
    Dataset,
    AMINO_ACIDS,
)


def _load_dataset() -> Tuple[List[str], NDArray[np.int_]]:
    sequences = np.loadtxt(
        f"{DIR_PATH}/data/dataset.csv", skiprows=1, usecols=0, delimiter=",", dtype=str
    ).tolist()
    labels = np.loadtxt(
        f"{DIR_PATH}/data/dataset.csv", skiprows=1, usecols=1, delimiter=",", dtype=int
    )
    return sequences, labels


def test_dense_layer() -> None:
    batch_size, d_in, d_layer, d_out = 7, 5, 3, 2
    model = _DenseLayer(d_in, d_layer, d_out, "relu")
    out = model(torch.rand((batch_size, d_in)))
    assert out.shape == torch.Size((batch_size, d_out))


def test_skipgram() -> None:
    (
        batch_size,
        d_sequence,
        d_embedding,
        d_layer,
        n_labels,
        backward_window,
        forward_window,
    ) = 17, 13, 11, 7, 5, 3, 2
    parameters = _SkipGramParameters(
        d_embedding=d_embedding,
        n_labels=n_labels,
        backward_window=backward_window,
        forward_window=forward_window,
        d_layer=d_layer,
        activation="relu",
    )
    model = _SkipGram(parameters)
    embedding = torch.rand((batch_size, d_sequence, d_embedding))
    labels = torch.randint(0, n_labels + 1, (batch_size, d_sequence))
    logits, _ = model(embedding, labels)
    expected_size = 0
    for i in range(batch_size):
        for j in range(d_sequence):
            if labels[i, j] == 0:
                continue
            for k in range(forward_window):
                target_index = j + k + 1
                if d_sequence <= target_index or labels[i, target_index] == 0:
                    continue
                expected_size += 1
            for k in range(backward_window):
                target_index = j - k - 1
                if target_index < 0 or labels[i, target_index] == 0:
                    continue
                expected_size += 1
    assert logits.shape == torch.Size((expected_size, n_labels))


def test_forward() -> None:
    """
    Test that a the Model class outputs predictions of the expected shape
    """
    batch_size = 2
    sequences, labels = _load_dataset()
    dataloader = DataLoader(
        Dataset(sequences, labels),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=Dataset.collate,
    )
    parameters = ModelParameters(pre_training=PreTrainingParameters())
    model = Model(parameters)
    batch_sequences, _ = next(iter(dataloader))

    logits, aa_labels = model(batch_sequences, pre_training=True)
    assert logits.shape == torch.Size((aa_labels.shape[0], len(AMINO_ACIDS) - 1))
    assert aa_labels.ndim == 1
    logits = model(batch_sequences, pre_training=False)
    assert logits.shape == torch.Size((batch_sequences.shape[0], 2))
