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


class QQPDataModule(HFDataModule):
    def __init__(self, num_labels, sentence_1_name, sentence_2_name, data_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.data_path = Path(data_path)

    def download_dataset(self):
        datasets.load_dataset("glue", "qqp")
        datasets.load_dataset(
            "csv",
            data_files={
                "train": str(self.data_path / "paws_train.tsv"),
                "val_test": str(self.data_path / "paws_dev_and_test.tsv"),
            },
            sep="\t",
        )

    def load_dataset(self):

        qqp = datasets.load_dataset("glue", "qqp")
        paws = datasets.load_dataset(
            "csv",
            data_files={
                "train": str(self.data_path / "paws_train.tsv"),
                "val_test": str(self.data_path / "paws_dev_and_test.tsv"),
            },
            sep="\t",
        )
        paws = paws.rename_column("sentence1", "question1")
        paws = paws.rename_column("sentence2", "question2")

        dataset = DatasetDict(
            {
                "train": qqp["train"],
                "validation": qqp["validation"],
                "test_paws": paws["val_test"],
            }
        )
        return dataset

    def test_dataloader(self):

        qqp_dataloader = DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=self.dataset["test_paws"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )
        return [qqp_dataloader]

    def get_test_names(self) -> "list[str]":
        return ["paws"]
