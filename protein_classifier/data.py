# Copyrigh: Lars Andersen Bratholm - 2024

"""
Data.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

# - for padding, . for start, | for end
AMINO_ACIDS = "-ARNDCQEGHILKMFPSTWYV.|"


class Dataset(torch.utils.data.Dataset):  # type: ignore
    """
    Data object for feeding datasets to pytorch Dataloader.
    """

    def __init__(
        self,
        sequences: List[str],
        labels: Union[List[int], List[bool], NDArray[np.int_], NDArray[np.bool_], None],
    ) -> None:
        """
        :param sequences: amino acid sequences
        :param label: labels
        """

        if labels is not None:
            assert len(sequences) == len(labels)
        self.size = len(sequences)
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        """
        The size of the dataset.
        """
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Tensor, Optional[int]]:
        """
        Get a sequence encoded as indices and target label from a given index.

        :param idx: index
        :returns: sequence and label
        """

        sequence = self.sequences[idx]
        sequence_indices = Dataset._amino_acid_char_to_index(sequence)
        if self.labels is None:
            label: Optional[int] = None
        else:
            label = int(self.labels[idx])
        return sequence_indices, label

    @staticmethod
    def _amino_acid_char_to_index(sequence: str) -> Tensor:
        """
        Translates an amino acid sequence to an array of indices.

        :param sequence: the amino acid sequence
        :returns: the encoded sequence
        """
        sequence_indices = torch.from_numpy(
            np.fromiter(
                (AMINO_ACIDS.index(char) for char in "." + sequence + "|"),
                dtype=np.int_,
            )
        )
        assert 0 not in sequence_indices
        return sequence_indices

    @staticmethod
    def collate(
        data: List[Tuple[Tensor, Optional[int]]],
    ) -> Tuple[Tensor, Union[Tensor, List[None]]]:
        """
        Collate function for the DataLoader that supports variable feature size by 0 padding.

        :param data: list of sequences and labels for each index in the batch
        :returns: concatenated padded sequences and targets
        """
        sequences = pad_sequence(
            [item[0] for item in data], batch_first=True, padding_value=0
        )
        labels_ = [item[1] for item in data]
        if None not in labels_:
            labels: Union[Tensor, List[None]] = torch.tensor(labels_, dtype=torch.int64)
        else:
            labels = labels_  # type: ignore[assignment]
        return sequences, labels
