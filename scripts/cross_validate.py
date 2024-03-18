#!/usr/bin/env python
# Copyright: Lars Andersen Bratholm - 2024

"""
Get 5-fold cross-validated predictions for a dataset
"""

import os
import sys
from typing import List, Literal

import numpy as np
import sklearn.model_selection
import tap
from loguru import logger
from numpy.typing import NDArray
from train_model import ArgumentParser as TrainerArgumentParser
from train_model import main as fit_model

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append(DIR_PATH + "../")

from protein_classifier.utils import configure_logger  # noqa: E402


class ArgumentParser(tap.Tap):
    """
    Get 5-fold cross-validated predictions for a dataset
    """

    data: str
    model_parameters: str
    training_parameters: str
    output: str
    num_workers: int
    accelerator: Literal["gpu", "cpu"]

    def configure(self) -> None:
        self.add_argument(
            "data", nargs="?", help="Location of the zarr store used for training data"
        )
        self.add_argument(
            "--model-parameters", "-m", nargs="?", help="Yaml file defining the Net"
        )
        self.add_argument(
            "--training-parameters",
            "-s",
            nargs="?",
            help="Yaml file defining the training settings",
        )
        self.add_argument("--output", "-o", nargs="?", help="The output folder.")
        self.add_argument(
            "--num-workers",
            "-w",
            nargs="?",
            help="The number of workers",
            default=0,
            type=int,
        )
        self.add_argument(
            "--accelerator",
            "-a",
            nargs="?",
            help="The accelerator ('gpu' or 'cpu')",
            default="gpu",
        )


def create_splits(args: ArgumentParser) -> List[NDArray[np.int_]]:
    """
    Create the 5 fold CV splits

    :param args: command-line arguments
    :returns: test_indices
    """
    np.random.seed(42)
    cv = sklearn.model_selection.KFold(n_splits=5, shuffle=True)
    sequences = np.loadtxt(
        f"{args.data}", skiprows=1, usecols=0, delimiter=",", dtype=str
    ).tolist()
    labels = np.loadtxt(f"{args.data}", skiprows=1, usecols=1, delimiter=",", dtype=int)
    indices = list(cv.split(labels))
    # TODO
    train_indices = [
        train_idx[: len(train_idx) // 2] for (train_idx, test_idx) in indices
    ]
    test_indices = [test_idx for (train_idx, test_idx) in indices]

    for i in range(5):
        output_folder = f"{args.output}/split_{i}"
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        save_csv(
            f"{output_folder}/training_data.csv",
            [sequences[j] for j in train_indices[i]],
            labels[train_indices[i]],
        )
        save_csv(
            f"{output_folder}/test_data.csv",
            [sequences[j] for j in test_indices[i]],
            labels[test_indices[i]],
        )
    return test_indices


def save_csv(filename: str, sequences: List[str], labels: NDArray[np.int_]) -> None:
    """
    Save sequences and labels to csv

    :param filename: the filename
    :param sequences: the sequences
    :param labels: the labels
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("amino_acid_sequence,label")
        for sequence, label in zip(sequences, labels, strict=True):
            f.write("\n")
            f.write(f"{sequence},{label}\n")


def create_trainer_args(  # pylint: disable=too-many-branches,too-many-statements
    args: ArgumentParser,
    split: int,
) -> TrainerArgumentParser:
    """
    Create the arguments used in the training script.

    :param args: command-line arguments
    :param split: the split index
    :returns: arguments for the trainer
    """
    # Create trainer args
    trainer_parser = TrainerArgumentParser()
    split_folder = f"{args.output}/split_{split}"
    trainer_input = [f"{split_folder}/training_data.csv"]
    trainer_input += ["--test-data", f"{split_folder}/test_data.csv"]
    trainer_input += ["--model-parameters", f"{args.model_parameters}"]
    trainer_input += ["--training-parameters", f"{args.training_parameters}"]
    trainer_input += ["--output", split_folder]
    trainer_input += ["--num-workers", str(args.num_workers)]
    trainer_input += ["--accelerator", args.accelerator]

    trainer_args = trainer_parser.parse_args(trainer_input)
    return trainer_args


def train_and_predict(
    args: ArgumentParser, test_indices: List[NDArray[np.int_]]
) -> None:
    """
    Train each model in the split and make predictions on the
    respective test sets.

    :param args: command-line arguments
    :param test_indices: the test indices
    """
    mean_accuracy = 0.0
    n_labels = sum(len(idx) for idx in test_indices)
    labels = np.zeros((n_labels, 2), dtype=np.int_)
    for i in range(5):
        trainer_args = create_trainer_args(args, i)
        accuracy = fit_model(trainer_args)
        mean_accuracy += accuracy / 5
        split_labels: NDArray[np.int_] = np.loadtxt(
            f"{args.output}/split_{i}/test_predictions.txt",
            skiprows=1,
            delimiter=",",
            dtype=int,
        )
        labels[test_indices[i]] = split_labels

    logger.info(f"Found CV mean accuracy of {mean_accuracy:.4g}")

    with open(f"{args.output}/accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{mean_accuracy:.4g}\n")

    with open(f"{args.output}/predictions.txt", "w", encoding="utf-8") as f:
        f.write("prediction,target")
        for i in range(n_labels):
            f.write("\n")
            f.write(f"{labels[i,0]},{labels[i,1]}")


def main(args: ArgumentParser) -> None:
    """
    Cross-validate the given dataset

    :param args: command-line arguments
    """
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    test_indices = create_splits(args)
    train_and_predict(args, test_indices)


if __name__ == "__main__":
    configure_logger(sink=None, log_level="INFO")
    parser = ArgumentParser()
    arguments = parser.parse_args()
    main(arguments)
