from argparse import ArgumentError
from typing import Optional, Tuple, Union

import os
import torch
import random
import datasets
from datasets.dataset_dict import DatasetDict
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import DataCollatorWithPadding
from src.datamodules.hf_datamodule import HFDataModule
from src.utils.utils import get_logger
from joblib.externals.loky.backend.context import get_context

log = get_logger(__name__)


class SNLIDataModule(HFDataModule):
    """SNLI Datamodule."""

    def __init__(self, num_labels, sentence_1_name, sentence_2_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def download_dataset(self):
        datasets.load_dataset("snli")

    def load_dataset(self):

        snli = datasets.load_dataset("snli")
        mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
        dataset = DatasetDict(
            {
                "train": snli["train"],
                "validation": snli["validation"],
                "test": snli["test"],
            }
        )
        return dataset

    def test_dataloader(self):

        test_dataloader = DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=self.dataset["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )
        return [test_dataloader]

    def get_test_names(self) -> "list[str]":
        return ["snli_test"]
