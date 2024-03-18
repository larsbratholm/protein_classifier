# Copyright Lars Andersen Bratholm - 2024

"""
Modules to classify function from sequences
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Literal

import numpy as np
import torch
from torch import Tensor, nn
from pydantic import BaseModel
import json

from .data import AMINO_ACIDS
from .utils import (
    name_to_activation,
)

# For caching some tensors.
MAX_SEQUENCE_LENGTH = 256


# NOTE: should use pydantic constrained integers for sanity
#       checking, but compatibility with mypy is tricky
class PreTrainingParameters(BaseModel):
    """
    Input parameters defining the pre-training method.

    :param backward_window: how far backwards in the sequence to make predictions
    :param forward_window: how far forward in the sequence to make predictions
    :param use_encoder: whether or not to use the encoder defined in ModelParameters
                        during pre-training.
    """

    backward_window: int = 2
    forward_window: int = 2
    use_encoder: bool = False


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

    :param d_embedding: the embedding size
    :param d_feedforward: the dimensionality of the hidden layer used
                          in the classifier
    :param activation: the activation function used in the classifier
    :param positional_encoding_linear_transform: add a linear transform in the
                                                 static positional encoding.
    :param positional_encoding_dropout: dropout in the positional encoding
    :param pre_training: pre-training parameters (no pre-training if None)
    :param encoder: parameters defining the encoder
    :param decoder: Optionally use a decoder with the given parameters to create
                    the embedding input to the classifier, instead of just using
                    the sequence "end" token.
    """

    d_embedding: int = 64
    d_feedforward: int = 128
    activation: str = "relu"
    positional_encoding_linear_transform: bool = False
    positional_encoding_dropout: float = 0.1
    pre_training: Optional[PreTrainingParameters] = None
    encoder: TransformerParameters = TransformerParameters()
    decoder: Optional[TransformerParameters] = None


class _SkipGramParameters(BaseModel):
    """
    Input parameters defining the _SkipGram method

    :param d_embedding: the size of the embedding
    :param n_labels: the number of unique labels to predict
    :param backward_window: how far backwards in the sequence to make predictions
    :param forward_window: how far forward in the sequence to make predictions
    :param d_layer: the hidden layer size used in the dense prediction layer
    :param activation: the activation function
    """

    d_embedding: int
    n_labels: int
    backward_window: int
    forward_window: int
    d_layer: int
    activation: str


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


class _SkipGram(nn.Module):
    """
    Skip-gram method for pre-training.
    Predicts the labels of amino acids in a range before or after
    each amino acid from the embedding of the central amino acid.
    """

    def __init__(self, parameters: _SkipGramParameters):
        """
        :param parameters: see definitions in _SkipGramParameters docs.
        """
        super().__init__()
        self.n_labels = parameters.n_labels
        self.forward_window = parameters.forward_window
        self.backward_window = parameters.backward_window
        n_dense_layers = self.forward_window + self.backward_window
        assert n_dense_layers > 0
        interaction_pairs = self._get_interaction_pairs()
        self.register_buffer("_interaction_pairs", interaction_pairs)
        self.dense_layers = torch.nn.ModuleList(
            [
                _DenseLayer(
                    parameters.d_embedding,
                    parameters.d_layer,
                    self.n_labels,
                    parameters.activation,
                )
                for _ in range(n_dense_layers)
            ]
        )

    def _get_interaction_pairs(self) -> Tensor:
        """
        Create the index-pairs of possible (source, target) labels
        given the window and max sequence length.

        :returns: (2, n_pairs) pairs
        """
        indices = torch.arange(MAX_SEQUENCE_LENGTH)
        forward_window = torch.arange(1, self.forward_window + 1)
        backward_window = torch.arange(-self.backward_window, 0)
        window = torch.cat([backward_window, forward_window])
        label_indices = indices[:, None] + window[None, :]
        source_indices = indices[:, None].repeat((1, window.shape[0]))
        mask = (label_indices >= 0) & (label_indices < MAX_SEQUENCE_LENGTH)

        pairs = torch.stack([source_indices[mask], label_indices[mask]])
        return pairs

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass

        :param embeddings: the input embeddings
        :param labels: the input labels
        :returns: target labels and their predicted logits
        """
        assert labels.shape == torch.Size((embeddings.shape[:2]))
        sequence_size = labels.shape[1]
        pair_indices = self._interaction_pairs[
            :, (self._interaction_pairs < sequence_size).all(0)
        ]
        embeddings_, target_labels, relative_positions = self._preprocess_data(
            labels, embeddings, pair_indices
        )
        logits = self._get_logits(embeddings_, relative_positions)

        return logits, target_labels

    def _preprocess_data(
        self, labels: Tensor, embeddings: Tensor, pair_indices: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Preprocess tensors to remove padding, and get the features/targets
        for classification

        :param labels: the input labels
        :param embeddings: the input embeddings
        :param pair_indices: the interacting feature/target pairs
        :returns: feature embeddings, target labels and their relative positions
        """
        batch_size = labels.shape[0]
        # Get all source and target pairs in the batch (remove padding)
        # Doing this in multiple steps for readability
        labels_ = labels[:, pair_indices.T]  # (batch_size, n_pairs, 2)
        labels_ = labels_.reshape(-1, 2)  # (batch_size * n_pairs, 2)
        padding_mask = (labels_ != 0).all(1)  # (batch_size * n_pairs, )
        labels_ = labels_[padding_mask, 1]  # (pairs_in_batch, )
        # Shift labels since we have removed 0-padding
        labels_ = labels_ - 1

        embeddings_ = embeddings[
            :, pair_indices[0]
        ]  # (batch_size, n_pairs, d_embedding)
        embeddings_ = embeddings_.view(
            -1, embeddings.shape[2]
        )  # (batch_size * n_pairs, d_embedding)
        embeddings_ = embeddings_[padding_mask, :]  # (pairs_in_batch, d_embedding)

        # Get the positional offset between source embedding and target label
        relative_positions = pair_indices[0] - pair_indices[1]
        relative_positions = relative_positions.repeat((batch_size, 1))
        relative_positions = relative_positions.view(-1)
        relative_positions = relative_positions[padding_mask]  # (pairs_in_batch, )

        return embeddings_, labels_, relative_positions

    def _get_logits(self, embeddings: Tensor, relative_positions: Tensor) -> Tensor:
        """
        Gather the logits as a different prediction layer is used based on
        the relative positions.

        :param embeddings: the input embeddings
        :param relative_positions: the relative positions
        :returns: the predicted logits
        """
        # Process the data through the layer corresponding the the positional
        # difference
        logits = embeddings.new_zeros(
            (embeddings.shape[0], self.n_labels), dtype=torch.float32
        )
        for i, layer in enumerate(self.dense_layers):
            if i < self.forward_window:
                mask = relative_positions == (i + 1)
            else:
                mask = relative_positions == -(
                    self.forward_window + self.backward_window - i
                )
            logits[mask] = layer(embeddings[mask]).to(logits.dtype)
        return logits


class Model(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    Classifier based on a transformer encoder. Either uses
    the "end" embedding as predictive features (which might be noisy due
    to the positional embedding), or the output of a transformer decoder.
    Supports pre-training with a skip-gram classifier.
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
        self._d_embedding = parameters.d_embedding

        # Embedding of amino acids
        token_embedding = nn.Embedding(
            len(AMINO_ACIDS), parameters.d_embedding, padding_idx=0
        )
        self.token_embedding = torch.jit.script(token_embedding)
        self.end_token_index = len(AMINO_ACIDS) - 1
        # Positional encoding
        positional_encoding = PositionalEncoding(
            parameters.d_embedding,
            parameters.positional_encoding_linear_transform,
            parameters.positional_encoding_dropout,
        )
        # Joint embedding
        embedding = nn.Sequential(token_embedding, positional_encoding)
        self.embedding = torch.jit.script(embedding)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            parameters.d_embedding,
            parameters.encoder.n_heads,
            parameters.encoder.d_feedforward,
            parameters.encoder.dropout,
            parameters.encoder.activation,
            batch_first=True,
        )
        encoder = nn.TransformerEncoder(  # type: ignore[no-untyped-call]
            encoder_layer,
            parameters.encoder.n_layers,
            nn.LayerNorm(parameters.d_embedding),
        )
        self._encoder = torch.jit.script(encoder)

        # Transformer decoder
        if parameters.decoder is not None:
            decoder_layer = nn.TransformerDecoderLayer(
                parameters.d_embedding,
                parameters.decoder.n_heads,
                parameters.decoder.d_feedforward,
                parameters.decoder.dropout,
                parameters.decoder.activation,
                batch_first=True,
            )
            decoder = nn.TransformerDecoder(  # type: ignore[no-untyped-call]
                decoder_layer,
                parameters.decoder.n_layers,
                nn.LayerNorm(parameters.d_embedding),
            )
            self._decoder = torch.jit.script(decoder)

        self._causal = (
            parameters.pre_training is not None
            and parameters.pre_training.backward_window == 0
        )
        if self._causal:
            self.register_buffer(
                "_causal_mask",
                nn.Transformer.generate_square_subsequent_mask(MAX_SEQUENCE_LENGTH),
            )

        if parameters.pre_training is None:
            self._pre_training = False
        if parameters.pre_training is not None:
            self._pre_training = True
            skip_gram_parameters = _SkipGramParameters(
                d_embedding=parameters.d_embedding,
                n_labels=len(AMINO_ACIDS) - 1,
                backward_window=parameters.pre_training.backward_window,
                forward_window=parameters.pre_training.forward_window,
                d_layer=parameters.d_feedforward,
                activation=parameters.activation,
            )
            skip_gram = _SkipGram(skip_gram_parameters)
            self.skip_gram = torch.jit.script(skip_gram)
            self._pre_training_use_encoder = parameters.pre_training.use_encoder

        classifier = _DenseLayer(
            parameters.d_embedding,
            parameters.d_feedforward,
            2,
            activation=parameters.activation,
        )
        self.classifier = torch.jit.script(classifier)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Initialize parameters
        """
        for parameter in self._encoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        if hasattr(self, "_decoder"):
            for parameter in self._decoder.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        sequences: Tensor,
        pre_training: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass. If pre_training is True, returns the predicted logits of parts
        of the sequence (skip-gram). Otherwise returns the predicted logits of
        the target labels.

        :param sequence: data features
        :param pre_training: whether or not to return pre-training logits instead of
                             target label logits
        :returns: predicted logits or (in case of pre-training) logits and target labels
        """
        batch_size, sequence_length = sequences.shape

        padding_mask = sequences == 0
        assert sequence_length < MAX_SEQUENCE_LENGTH
        if pre_training is True:
            assert self._pre_training is True
            if self._causal is True:
                causal_mask: Optional[Tensor] = self._causal_mask[
                    :sequence_length, :sequence_length
                ]
            else:
                causal_mask = None
            if self._pre_training_use_encoder is True:
                embedding = self.embedding(sequences)
                embedding = self._encoder(
                    embedding,
                    mask=causal_mask,
                    src_key_padding_mask=padding_mask,
                    is_causal=self._causal,
                )
            else:
                embedding = self.token_embedding(sequences)
            predictions: Tuple[Tensor, Tensor] = self.skip_gram(embedding, sequences)
            return predictions

        embedding = self.embedding(sequences)
        embedding = self._encoder(embedding, src_key_padding_mask=padding_mask)

        if hasattr(self, "_decoder"):
            final_embedding = self._decoder(
                embedding.new_ones((batch_size, 1, self._d_embedding))
                / self._d_embedding,
                embedding,
                memory_key_padding_mask=padding_mask,
            )[:, -1]
        else:
            final_embedding = embedding[sequences == self.end_token_index]

        logits: Tensor = self.classifier(final_embedding)
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
