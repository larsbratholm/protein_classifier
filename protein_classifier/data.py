# Copyrigh: Lars Andersen Bratholm - 2024

"""
Data.
"""

from __future__ import annotations

from typing import Optional, Tuple
import os

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .utils import load_vocabulary

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"

# - for padding, . for start, | for end
AMINO_ACIDS = "-ARNDCQEGHILKMFPSTWYVXO.|"
V_GENE_VOCAB = load_vocabulary(f"{DIR_PATH}vocabularies/V.GENE")
J_GENE_VOCAB = load_vocabulary(f"{DIR_PATH}vocabularies/J.GENE")
LABEL_VOCAB = load_vocabulary(f"{DIR_PATH}vocabularies/EPITOPE.SPECIES")


def load_dataset(location: str) -> NDArray[np.str_]:
    """
    Load datasets.

    :param location: the location of the csv file to parse
    :returns: dataset
    """
    data = np.loadtxt(location, skiprows=1, delimiter=",", dtype=str).T
    return data


class Dataset(torch.utils.data.Dataset):  # type: ignore
    """
    Data object for feeding datasets to pytorch Dataloader.
    """

    def __init__(
        self,
        sequences: NDArray[np.str_],
        v_gene: NDArray[np.str_],
        j_gene: NDArray[np.str_],
        labels: NDArray[np.str_] | None,
        pad_size: int = 256,
    ) -> None:
        """
        :param sequences: amino acid sequences
        :param v_gene: the v_gene
        :param j_gene: the j_gene
        :param label: labels
        :param pad_size: the size to pad sequence token indices to
        """
        if labels is not None:
            assert len(sequences) == len(labels)
        self.size = len(sequences)
        self.sequences = sequences
        self.v_gene = v_gene
        self.j_gene = j_gene
        self.labels = labels
        self.pad_size = pad_size

    def __len__(self) -> int:
        """
        The size of the dataset.
        """
        return self.size

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, int, Optional[int]]:
        """
        Get a sequence encoded as indices, V.GENE, J.GENE and target label from a given index.

        :param idx: index
        :returns: sequence, v_gene and label
        """
        sequence = self.sequences[idx]
        sequence_indices = self._amino_acid_char_to_index(sequence)
        v_gene_index = V_GENE_VOCAB.index(self.v_gene[idx])
        j_gene_index = J_GENE_VOCAB.index(self.j_gene[idx])

        if self.labels is None:
            label: Optional[int] = None
        else:
            label = LABEL_VOCAB.index(self.labels[idx])
        return sequence_indices, v_gene_index, j_gene_index, label

    def _amino_acid_char_to_index(self, sequence: str) -> Tensor:
        """
        Translates an amino acid sequence to a padded array of indices.

        :param sequence: the amino acid sequence
        :returns: the padded token indices
        """
        padded_indices = torch.zeros(self.pad_size, dtype=torch.int64)
        padded_indices[: len(sequence) + 2] = torch.from_numpy(
            np.fromiter(
                (AMINO_ACIDS.index(char) for char in "." + sequence + "|"),
                dtype=np.int_,
            )
        )
        return padded_indices


#    @staticmethod
#    def collate(
#        data: List[Tuple[Tensor, Optional[int]]],
#    ) -> Tuple[Tensor, Union[Tensor, List[None]]]:
#        """
#        Collate function for the DataLoader that supports variable feature size by 0 padding.
#
#        :param data: list of sequences and labels for each index in the batch
#        :returns: concatenated padded sequences and targets
#        """
#        sequences = pad_sequence(
#            [item[0] for item in data], batch_first=True, padding_value=0
#        )
#        v_genes = torch.tensor([item[1] for item in data], dtype=torch.int64)
#        j_genes = torch.tensor([item[2] for item in data], dtype=torch.int64)
#        labels_ = [item[3] for item in data]
#        if None not in labels_:
#            labels: Union[Tensor, List[None]] = torch.tensor(labels_, dtype=torch.int64)
#        else:
#            labels = labels_  # type: ignore[assignment]
#        return sequences, v_genes, j_genes, labels
