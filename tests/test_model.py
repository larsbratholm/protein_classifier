# Copyright: Lars Andersen Bratholm - 2024

"""
Tests for models
"""

import os
import sys

import torch
from torch.utils.data import DataLoader

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append(DIR_PATH + "../")


from protein_classifier.models import (  # noqa:E402
    Model,
    _DenseLayer,
    ModelParameters,
)
from protein_classifier.data import (  # noqa:E402
    Dataset,
    load_dataset,
)


def test_dense_layer() -> None:
    batch_size, d_in, d_layer, d_out = 7, 5, 3, 2
    model = _DenseLayer(d_in, d_layer, d_out, "relu")
    out = model(torch.rand((batch_size, d_in)))
    assert out.shape == torch.Size((batch_size, d_out))


def test_forward() -> None:
    """
    Test that a the Model class outputs predictions of the expected shape
    """
    batch_size = 2
    sequences, v_genes, j_genes, labels = load_dataset(f"{DIR_PATH}/data/dataset.csv")
    dataloader = DataLoader(
        Dataset(sequences, v_genes, j_genes, labels),
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=Dataset.collate,
    )
    parameters = ModelParameters()
    model = Model(parameters)
    batch_sequences, batch_v_genes, batch_j_genes, _ = next(iter(dataloader))

    logits = model(batch_sequences, batch_v_genes, batch_j_genes)
    assert logits.shape == torch.Size((batch_sequences.shape[0], parameters.n_classes))
