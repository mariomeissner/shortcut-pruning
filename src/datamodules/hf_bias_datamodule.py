from argparse import ArgumentError
from typing import Optional, Tuple

import os
from datasets.dataset_dict import DatasetDict
import torch
import datasets
from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import DataCollatorWithPadding
from src.utils import utils
from bias_utils import load_bias, load_teacher_probs

log = utils.get_logger(__name__)


class HFDataModule(LightningDataModule):
    """
    LightningDataModule for HF Datasets.
    Requires a pre-processed (tokenized, cleaned...) dataset provided within the `data` folder.
    Might require adjustments if your dataset doesn't follow the structure of SNLI or MNLI.

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
        data_dir: str,
        dataset_name: str,
        bias_name: str,
        tokenizer_name: str,
        batch_size: int = 64,
        max_length: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = None
        self.tokenizer = None
        self.collator_fn = None

        self.dataset_path = Path(self.hparams.data_dir) / self.hparams.dataset_name
        self.bias_path = Path(self.hparams.data_dir) / self.hparams.bias_name

        self.eval_key = "validation"
        self.test_key = "test"

        if "mnli" in self.hparams.dataset_name:
            self.eval_key += "_matched"
            self.test_key += "_matched"

        self.keep_columns = [
            "idx",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "bias",
            "teacher_probs",
        ]

    @property
    def num_classes(self) -> int:
        return 3

    def prepare_data(self):
        """
        We should not assign anything here, so this function simply ensures
        that the pre-processed data is available.
        """
        if not os.path.exists(self.dataset_path):
            raise ValueError("The provided folder does not exist.")
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name, use_fast=True)  # TODO: Load according to model-name

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        if not self.tokenizer:
            # TODO: Load according to model-name
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name, use_fast=True)

        if not self.collator_fn:
            self.collator_fn = DataCollatorWithPadding(tokenizer=self.tokenizer)

        if not self.dataset:
            log.info("Preparing the dataset and appending biases.")
            self.dataset = datasets.load_from_disk(self.dataset_path)
            
            # Get rid of mismatched
            self.dataset = DatasetDict({
                "train" : self.dataset['train'],
                "validation_matched" : self.dataset['validation_matched'],
                "test_matched" : self.dataset['test_matched']
            })

            bias = load_teacher_probs(self.bias_path)

            def append_bias(example):
                example["teacher_probs"] = bias[str(example["idx"])]
                return example

            def append_empty_bias(example):
                example["teacher_probs"] = [0,0,0] # Dummy bias
                return example

            self.dataset["train"] = self.dataset["train"].map(append_bias)
            self.dataset[self.eval_key] = self.dataset[self.eval_key].map(append_empty_bias)
            self.dataset[self.test_key] = self.dataset[self.test_key].map(append_empty_bias)

            keep_columns = [column for column in self.keep_columns if column in self.dataset["train"].column_names]
            self.dataset.set_format("torch", columns=keep_columns)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset[self.eval_key],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset[self.test_key],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )
