# Copyright Lars Andersen Bratholm - 2024

"""
Modules to classify function from sequences and genes
"""

from __future__ import annotations

from typing import Tuple, Union, Literal

import numpy as np
import torch
from torch import Tensor, nn
from pydantic import BaseModel
import json

from .data import AMINO_ACIDS, V_GENE_VOCAB, J_GENE_VOCAB
from .utils import (
    name_to_activation,
)

# For caching some tensors.
MAX_SEQUENCE_LENGTH = 256


class TransformerParameters(BaseModel):
    """
    Input parameters defining the transformer encoder/decoder.

    :param n_heads: the number of attention heads
    :param n_layers: the number of encoder/decoder layers
    :param d_feedforward: the dimensionality of the feedforward layer
    :param dropout: the dropout
    :param activation: the activation function in the feedforward layer
    """

    n_heads: int = 1
    n_layers: int = 2
    d_feedforward: int = 128
    dropout: float = 0.1
    activation: Literal["relu", "gelu"] = "gelu"


class ModelParameters(BaseModel):
    """
    Input parameters defining the Model

    :param n_classes: the number of classlabels to support
    :param d_sequence_embedding: the embedding size of amino acids
    :param d_gene_embedding: the embedding size of genes.
    :param d_feedforward: the dimensionality of the hidden layer used
                          in the classifier
    :param activation: the activation function used in the classifier
    :param positional_encoding_linear_transform: add a linear transform in the
                                                 static positional encoding.
    :param positional_encoding_dropout: dropout in the positional encoding
    :param encoder: parameters defining the encoder
    """

    n_classes: int = 3
    d_sequence_embedding: int = 64
    d_gene_embedding: int = 32
    d_feedforward: int = 128
    activation: str = "relu"
    positional_encoding_linear_transform: bool = False
    positional_encoding_dropout: float = 0.1
    encoder: TransformerParameters = TransformerParameters()


class PositionalEncoding(nn.Module):
    """
    From pytorch tutorial with minor variations.
    """

    def __init__(
        self, d_embedding: int, linear_layer: bool = False, dropout: float = 0.1
    ):
        """
        :param d_embedding: the size of the embedding
        :param linear_layer: whether or not to include a linear transform of the
                             static embedding
        :param dropout: the dropout
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(MAX_SEQUENCE_LENGTH, d_embedding)
        position = torch.arange(0, MAX_SEQUENCE_LENGTH, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_embedding, 2).float() * (-np.log(10000.0) / d_embedding)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        if linear_layer is True:
            self.linear = nn.Linear(d_embedding, d_embedding)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add the positional embedding to the input

        :param x: the input embedding
        :returns: the embedding with encoded position
        """
        pe = self.pe[:, : x.shape[1], :]
        if hasattr(self, "linear"):
            pe = self.linear(pe)
        x = x + pe
        x = self.dropout(x)
        return x


class _DenseLayer(nn.Module):
    """
    Dense layer with layer norm
    """

    def __init__(self, d_in: int, d_layer: int, d_out: int, activation: str):
        """
        :param d_in: the input size
        :param d_layer: the size of the hidden layer
        :param d_out: the output size
        :param activation: the activation function
        """
        super().__init__()
        activation_function = name_to_activation(activation, {})
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_layer, bias=False),
            torch.nn.LayerNorm(d_layer),
            activation_function,
            torch.nn.Linear(d_layer, d_out, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass
        """
        out: Tensor = self.dense(x)
        return out


class Model(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    Classifier based on a transformer encoder. Uses the "start" token
    embedding as predictive features, together with learnt embeddings
    for the genes as input for a dense classifier.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        parameters: ModelParameters = ModelParameters(),
    ) -> None:
        """
        :param parameters: see ModelParameters docs.
        """
        super().__init__()

        # Store input arguments for save/load
        self._model_parameters = parameters.model_dump_json()
        self._d_sequence_embedding = parameters.d_sequence_embedding
        self._d_gene_embedding = parameters.d_gene_embedding

        # Embedding of amino acids
        token_embedding = nn.Embedding(
            len(AMINO_ACIDS), parameters.d_sequence_embedding, padding_idx=0
        )
        # Positional encoding
        positional_encoding = PositionalEncoding(
            parameters.d_sequence_embedding,
            parameters.positional_encoding_linear_transform,
            parameters.positional_encoding_dropout,
        )
        # Joint embedding
        self.embedding = nn.Sequential(token_embedding, positional_encoding)

        # Gene embedding
        self.v_gene_embedding = nn.Embedding(
            len(V_GENE_VOCAB), parameters.d_gene_embedding
        )
        self.j_gene_embedding = nn.Embedding(
            len(J_GENE_VOCAB), parameters.d_gene_embedding
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            parameters.d_sequence_embedding,
            parameters.encoder.n_heads,
            parameters.encoder.d_feedforward,
            parameters.encoder.dropout,
            parameters.encoder.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            parameters.encoder.n_layers,
            nn.LayerNorm(parameters.d_sequence_embedding),
            enable_nested_tensor=False,
        )

        self.classifier = _DenseLayer(
            parameters.d_sequence_embedding + 2 * parameters.d_gene_embedding,
            parameters.d_feedforward,
            parameters.n_classes,
            activation=parameters.activation,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize parameters
        """
        for parameter in self.encoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        sequences: Tensor,
        v_genes: Tensor,
        j_genes: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass. Returns the predicted logits of the target labels.

        :param sequence: amino acid token indices
        :param v_genes: variable gene token indices
        :param j_genes: joining gene token indices
        :returns: predicted logits
        """
        batch_size, sequence_length = sequences.shape

        padding_mask = sequences == 0
        assert sequence_length <= MAX_SEQUENCE_LENGTH

        embedding = self.embedding(sequences)
        encoding = self.encoder(embedding, src_key_padding_mask=padding_mask)

        v_gene_embedding = self.v_gene_embedding(v_genes)
        j_gene_embedding = self.j_gene_embedding(j_genes)

        stacked = [encoding[:, 0], v_gene_embedding, j_gene_embedding]
        features = torch.cat(stacked, dim=1)

        logits: Tensor = self.classifier(features)
        return logits

    @torch.jit.ignore  # type: ignore[misc]
    def save_model(
        self,
        filename: str,
    ) -> None:
        """
        Save the model.

        :param filename: model filename
        """
        store = {"parameters": self._model_parameters, "state_dict": self.state_dict()}
        torch.save(store, filename)

    @staticmethod
    def load_model(filename: str) -> Model:
        """
        Load saved model

        :param filename: model filename
        :returns: initialized model with loaded structure and parameters
        """

        store = torch.load(filename, map_location="cpu")
        parameters = ModelParameters.parse_raw(json.dumps(store["parameters"]))

        model = Model(parameters)
        model.load_state_dict(store["state_dict"])
        return model
