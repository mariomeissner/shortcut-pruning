import os
import random
import math
from argparse import ArgumentError
from pathlib import Path
from typing import Optional, Tuple, Union

import datasets
import torch
from datasets.dataset_dict import DatasetDict
from joblib.externals.loky.backend.context import get_context
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from transformers import DataCollatorWithPadding
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.utils.bias_utils import load_bias_probs
from src.utils.utils import get_logger

log = get_logger(__name__)


class HFDataModule(LightningDataModule):
    """
    Base LightningDataModule for Huggingface Datasets.
    Requires a pre-processed (tokenized, cleaned...) dataset provided within the `data` folder.
    Might require adjustments if your dataset doesn't follow the structure of SNLI or MNLI.
    """

    def __init__(
        self,
        tokenizer_name: str,
        select_train_samples: int = 0,
        bias_path: str = None,
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

        self.multiprocessing_context = get_context("loky") if num_workers > 1 else None

        self.keep_columns = [
            "idx",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "bias",
            "bias_probs",
        ]

    @property
    def num_classes(self) -> int:
        return self.hparams.num_labels

    @staticmethod
    def process_data(dataset: DatasetDict, keep_columns, hparams, tokenizer):
        """To be called in setup."""
        log.info("Processing dataset.")

        # Rename label to labels for consistency
        for key in dataset.keys():
            if "label" in dataset[key].column_names:
                dataset[key] = dataset[key].rename_column("label", "labels")

        # Apply tokenization and filter out invalid labels
        fn_kwargs = {
            "sentence_1_name": hparams.sentence_1_name,
            "sentence_2_name": hparams.sentence_2_name,
            "max_length": hparams.max_length,
            "tokenizer": tokenizer,
        }
        dataset = dataset.map(HFDataModule.map_func, batched=True, num_proc=1, fn_kwargs=fn_kwargs)
        # dataset = dataset.filter(lambda sample: sample["labels"] != -1)

        # Append bias if provided
        if hparams.bias_path:
            if hparams.bias_path == "random":
                log.info("Appending random probability vectors.")
                special = "random"
                bias_dict = None
            elif hparams.bias_path == "flat":
                log.info("Appending flat probability vectors.")
                special = "flat"
                bias_dict = None
            else:
                special = False
                log.info(f"Appending bias from file {hparams.bias_path}.")
                bias_dict = load_bias_probs(hparams.bias_path)

            dataset["train"] = dataset["train"].map(
                HFDataModule.append_bias,
                fn_kwargs={
                    "bias_dict": bias_dict,
                    "special": special,
                    "num_labels": hparams.num_labels,
                },
            )

        # Set torch format
        keep_columns = [column for column in keep_columns if column in dataset["train"].column_names]
        dataset["train"].set_format("torch", columns=keep_columns, output_all_columns=False)

        if "idx" in keep_columns:
            keep_columns.remove("idx")
        if "bias_probs" in keep_columns:
            keep_columns.remove("bias_probs")

        for key in dataset.keys():
            if key != "train":
                dataset[key].set_format("torch", columns=keep_columns, output_all_columns=False)
            # remove_cols = [column for column in dataset[key].column_names if not column in keep_columns]
            # dataset[key].remove_columns_(remove_cols)
        return dataset

    @staticmethod
    def map_func(example_batch, sentence_1_name, sentence_2_name, tokenizer, max_length):
        """To be called by datasets.map"""
        if sentence_2_name is None:
            sents = (example_batch[sentence_1_name],)
        else:
            sents = (example_batch[sentence_1_name], example_batch[sentence_2_name])
        result = tokenizer(*sents, max_length=max_length, truncation="longest_first")
        return result

    @staticmethod
    def append_bias(example, bias_dict, special=None, num_labels=3):
        if special == "random":
            random_vector = [random.random() for _ in range(num_labels)]
            vector_sum = sum(random_vector)
            bias = [value / vector_sum for value in random_vector]
        elif special == "flat":
            bias = [1 / num_labels for _ in range(num_labels)]
        else:
            bias = bias_dict[str(example["idx"])]
        # assert math.isclose(sum(bias), 1.0)
        example["bias_probs"] = bias
        return example

    def prepare_data(self):
        """
        Required datamodule function.
        We should not assign anything here, so this function simply caches the tokenizer and dataset.
        """
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name, use_fast=True)
        self.download_dataset()

    def load_dataset(self):
        raise NotImplementedError()

    def download_dataset(self):
        raise NotImplementedError()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before `trainer.fit()` or `trainer.test()`."""

        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name, use_fast=True)

        if not self.collator_fn:
            self.collator_fn = DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=8)

        if not self.dataset:
            self.dataset = self.load_dataset()

            # Select training samples if specificed
            if self.hparams.select_train_samples:
                # datasets.Dataset slicing returns a dict
                train_dict = self.dataset["train"].shuffle()[: self.hparams.select_train_samples]
                self.dataset["train"] = datasets.Dataset.from_dict(train_dict)

            self.dataset = self.process_data(
                self.dataset,
                self.keep_columns,
                tokenizer=self.tokenizer,
                hparams=self.hparams,
            )

    def get_test_names(self) -> "list[str]":
        raise NotImplementedError

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=self.dataset["train"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=shuffle,
        )
    def minitrain_dataloader(self, shuffle=True):
        return DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=datasets.Dataset.from_dict(self.dataset["train"][:1000]),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=self.dataset["validation"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        raise NotImplementedError()
