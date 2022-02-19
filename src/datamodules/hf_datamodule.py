from argparse import ArgumentError
from typing import Optional, Tuple, Union

import os
import torch
import datasets
from datasets.dataset_dict import DatasetDict
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import DataCollatorWithPadding
from src.utils.utils import get_logger
from joblib.externals.loky.backend.context import get_context

log = get_logger(__name__)


class HFDataModule(LightningDataModule):
    """
    LightningDataModule for HF Datasets.
    Requires a pre-processed (tokenized, cleaned...) dataset provided within the `data` folder.
    Might require adjustments if your dataset doesn't follow the structure of SNLI or MNLI.

    Pass a (group, set_name) tuple to `dataset_name` if your dataset is part of a group such as glue.
    For example, for MNLI, use ('glue', 'mnli').

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        num_labels: int,
        tokenizer_name: str,
        sentence_1_name: str,
        sentence_2_name: str,
        batch_size: int = 64,
        max_length: int = 128,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()

        # Explicitly allow tokenizers to parallelize
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = None
        self.tokenizer = None
        self.collator_fn = None

        self.keep_columns = [
            "idx",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
        ]

    @property
    def num_classes(self) -> int:
        return self.hparams.num_labels

    @staticmethod
    def process_data(dataset: DatasetDict, keep_columns, hparams, tokenizer):
        """To be called in setup."""
        log.info("Processing dataset.")
        # Rename label to labels for consistency
        if "label" in dataset.column_names["train"]:
            dataset = dataset.rename_column("label", "labels")

        # Apply tokenization and filter out invalid labels
        fn_kwargs = {
            "sentence_1_name": hparams.sentence_1_name,
            "sentence_2_name": hparams.sentence_2_name,
            "tokenizer": tokenizer,
            "max_length": hparams.max_length,
        }
        dataset = dataset.map(HFDataModule.map_func, batched=True, num_proc=4, fn_kwargs=fn_kwargs)
        dataset = dataset.filter(lambda sample: sample["labels"] != -1)

        # Remove all unnecessary columns
        keep_columns = [column for column in keep_columns if column in dataset["train"].column_names]

        # Set torch format
        dataset.set_format("torch", columns=keep_columns)
        return dataset

    @staticmethod
    def map_func(example_batch, sentence_1_name, sentence_2_name, tokenizer, max_length):
        """To be called by datasets.map"""
        # TODO: Allow support for single sentence
        sents = (example_batch[sentence_1_name], example_batch[sentence_2_name])
        result = tokenizer(*sents, max_length=max_length, truncation="longest_first")
        return result

    def prepare_data(self):
        """
        Required datamodule function.
        We should not assign anything here, so this function simply caches the tokenizer and dataset.
        """
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name, use_fast=True)
        self.download_dataset()

    def download_dataset(self):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before `trainer.fit()` or `trainer.test()`."""

        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name, use_fast=True)

        if not self.collator_fn:
            self.collator_fn = DataCollatorWithPadding(tokenizer=self.tokenizer)

        if not self.dataset:
            self.dataset = self.download_dataset()
            self.dataset = self.process_data(
                self.dataset,
                self.keep_columns,
                tokenizer=self.tokenizer,
                hparams=self.hparams,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=True,
            multiprocessing_context=get_context("loky"),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset["validation"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
            multiprocessing_context=get_context("loky"),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
            multiprocessing_context=get_context("loky"),
        )
