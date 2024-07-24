#!/usr/bin/env python
# Copyright: Lars Andersen Bratholm - 2024

"""
Runs inference on the given data with the given model
"""

import os
import sys
from typing import Optional, Literal

import numpy as np

import pytorch_lightning as pl
import tap
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader

DIR_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append(DIR_PATH + "../")

from protein_classifier.trainer import LightningModel  # noqa:E402
from protein_classifier.utils import (  # noqa:E402
    configure_logger,
)
from protein_classifier.data import Dataset, load_dataset, LABEL_VOCAB  # noqa:E402


class ArgumentParser(tap.Tap):
    """
    Run inference
    """

    data: str
    checkpoint: str
    output: str
    batch_size: int
    num_workers: int
    accelerator: Literal["gpu", "cpu"]

    def configure(self) -> None:
        self.add_argument(
            "data",
            nargs="?",
            help="Location of the data csv file",
        )
        self.add_argument("checkpoint", nargs="?", help="Model checkpoint to use")
        self.add_argument(
            "--output", "-o", nargs="?", default="./", help="The output folder."
        )
        self.add_argument(
            "--batch-size",
            "-b",
            nargs="?",
            help="The batch size",
            default=512,
            type=int,
        )
        self.add_argument(
            "--num-workers",
            "-w",
            nargs="?",
            help="The number of workers for the dataloader",
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


def create_dataloader(
    args: ArgumentParser,
) -> DataLoader[tuple[Tensor, Tensor, Tensor, Optional[Tensor]]]:
    """
    Create the data loader

    :param args: command-line arguments
    :returns: dataloader
    """
    logger.info("Creating dataloader")
    data = load_dataset(f"{args.data}")
    max_sequence_length = max(len(item) for item in data[0])

    num_workers = (
        len(os.sched_getaffinity(0)) if args.num_workers == 0 else args.num_workers
    )
    dataloader = DataLoader(
        Dataset(*data, max_sequence_length + 2),  # type: ignore[call-arg]
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )
    return dataloader


def run_inference(
    model: LightningModel,
    dataloader: DataLoader[tuple[Tensor, Tensor, Tensor, Optional[Tensor]]],
    output_folder: str,
    accelerator: Literal["gpu", "cpu"] = "gpu",
) -> None:
    """
    Make predictions on the data set, and compute accuracy

    :param model: the model
    :param dataloader: the test dataloader
    :param output_folder: the output folder
    """
    logger.info("Starting inference")
    trainer = pl.Trainer(
        accelerator=accelerator,
        enable_progress_bar=True,
        default_root_dir=f"{output_folder}",
    )
    predictions = np.concat(trainer.predict(model, dataloader))  # type: ignore[arg-type]
    labels = np.concat([batch[3] for batch in dataloader])
    accuracy = (predictions == labels).sum().item() / len(labels)
    with open(f"{output_folder}/accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy:.4g}\n")
    with open(f"{output_folder}/predictions.txt", "w", encoding="utf-8") as f:
        f.write("prediction,target\n")
        for i in range(len(labels)):
            prediction_label = LABEL_VOCAB[int(predictions[i].item())]
            target_label = LABEL_VOCAB[int(labels[i].item())]
            f.write(f"{prediction_label},{target_label}\n")


def main(args: ArgumentParser) -> None:
    """
    Runs inference with the given data and model

    :param args: command-line arguments
    """
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    configure_logger(sink=f"{args.output}/log.txt", log_level="DEBUG", mode="w+")

    dataloader = create_dataloader(args)
    model = LightningModel.load_from_checkpoint(checkpoint_path=args.checkpoint)
    run_inference(model, dataloader, args.output, accelerator=args.accelerator)


if __name__ == "__main__":
    parser = ArgumentParser()
    arguments = parser.parse_args()
    main(arguments)
