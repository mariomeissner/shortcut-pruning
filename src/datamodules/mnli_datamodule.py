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


class MNLIDataModule(HFDataModule):
    """MNLI Datamodule."""

    def __init__(self, num_labels, sentence_1_name, sentence_2_name, data_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.data_path = Path(data_path)

    def download_dataset(self):
        datasets.load_dataset("glue", "mnli")
        datasets.load_dataset("hans")
        datasets.load_dataset("csv", data_files=str(self.data_path / "dev_matched_hard.tsv"), sep="\t")
        datasets.load_dataset("csv", data_files=str(self.data_path / "dev_mismatched_hard.tsv"), sep="\t")

    def load_dataset(self):

        mnli = datasets.load_dataset("glue", "mnli")
        hans = datasets.load_dataset("hans")
        remove_cols = [
            "index1",
            "index2",
            "index3",
            "genre",
            "premise_parse",
            "hypothesis_parse",
            "premise_gloss",
            "hypothesis_gloss",
            "label1",
            "label2",
            "label3",
            "label4",
            "label5",
        ]
        m_hard = datasets.load_dataset("csv", data_files=str(self.data_path / "dev_matched_hard.tsv"), sep="\t")
        mm_hard = datasets.load_dataset("csv", data_files=str(self.data_path / "dev_mismatched_hard.tsv"), sep="\t")
        m_hard.remove_columns_(remove_cols)
        mm_hard.remove_columns_(remove_cols)
        mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
        m_hard = m_hard.map(lambda sample: {"labels": mapping[sample["labels"]]})
        mm_hard = mm_hard.map(lambda sample: {"labels": mapping[sample["labels"]]})
        dataset = DatasetDict(
            {
                "train": mnli["train"],
                "validation": mnli["validation_matched"],
                "test_hans": hans["validation"],
                "test_m_hard": m_hard["train"],
                "test_mm_hard": mm_hard["train"],
            }
        )
        return dataset

    def test_dataloader(self):

        hans_dataloader = DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=self.dataset["test_hans"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )
        m_hard_dataloader = DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=self.dataset["test_m_hard"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )
        mm_hard_dataloader = DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=self.dataset["test_mm_hard"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )
        return [hans_dataloader, m_hard_dataloader, mm_hard_dataloader]

    def get_test_names(self) -> "list[str]":
        return ["hans", "m_hard", "mm_hard"]
