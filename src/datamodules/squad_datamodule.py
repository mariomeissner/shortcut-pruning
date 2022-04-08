from argparse import ArgumentError
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import os
import torch
import random
import datasets
from src.datamodules import squad_processing
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


class SquadDatamodule(LightningDataModule):
    def __init__(
        self,
        output_dir: str,
        tokenizer_name: str,
        n_best_size: int,
        doc_stride: int,
        max_answer_length: int,
        version_2_with_negative: bool,
        null_score_diff_threshold: float,
        padding: str = "max_length",
        select_train_samples: int = 0,
        batch_size: int = 16,
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
        self.val_example_id_strings = OrderedDict()
        self.test1_example_id_strings = OrderedDict()
        self.test2_example_id_strings = OrderedDict()

        self.multiprocessing_context = get_context("loky") if num_workers > 1 else None

        self.keep_columns = [
            "idx",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            # "labels",
        ]

    @staticmethod
    def process_data(
        dataset: DatasetDict,
        keep_columns,
        hparams,
        tokenizer,
        val_example_id_strings,
        test1_example_id_strings,
        test2_example_id_strings,
    ):
        """To be called in setup."""
        log.info("Processing dataset.")

        # Apply tokenization and filter out invalid labels
        fn_kwargs = {
            "tokenizer": tokenizer,
            "max_length": hparams.max_length,
            "pad_on_right": tokenizer.padding_side == "right",
            "question_column_name": "question",
            "context_column_name": "context",
            "answer_column_name": "answers",
            "doc_stride": hparams.doc_stride,
            "padding": hparams.padding,
        }

        # Remove all unnecessary columns
        # keep_columns = [column for column in keep_columns if column in dataset["train"].column_names]

        prepare_train_features = partial(squad_processing.prepare_train_features, **fn_kwargs)
        dataset["train"] = dataset["train"].map(
            prepare_train_features,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
        )
        fn_kwargs.pop("answer_column_name")
        # prepare_validation_features = partial(
        #     squad_processing.prepare_validation_features, example_id_strings=example_id_strings, **fn_kwargs
        # )
        dataset["validation_original"] = dataset["validation"]  # keep an original copy for computing metrics
        dataset["validation"] = dataset["validation"].map(
            partial(squad_processing.prepare_validation_features, example_id_strings=val_example_id_strings, **fn_kwargs),
            load_from_cache_file=False,
            batched=True,
            num_proc=1,
            remove_columns=dataset["validation"].column_names,
        )
        dataset["test_addsent_original"] = dataset["test_addsent"]  # keep an original copy for computing metrics
        dataset["test_addsent"] = dataset["test_addsent"].map(
            partial(squad_processing.prepare_validation_features, example_id_strings=test1_example_id_strings, **fn_kwargs),
            load_from_cache_file=False,
            batched=True,
            num_proc=1,
            remove_columns=dataset["test_addsent"].column_names,
        )
        dataset["test_addonesent_original"] = dataset["test_addonesent"]  # keep an original copy for computing metrics
        dataset["test_addonesent"] = dataset["test_addonesent"].map(
            partial(squad_processing.prepare_validation_features, example_id_strings=test2_example_id_strings, **fn_kwargs),
            load_from_cache_file=False,
            batched=True,
            num_proc=1,
            remove_columns=dataset["test_addonesent"].column_names,
        )
        # Set torch format
        # dataset["train"].set_format("torch")
        # dataset["validation"].set_format("torch")
        return dataset

    def prepare_data(self):
        """
        Required datamodule function.
        We should not assign anything here, so this function simply caches the tokenizer and dataset.
        """
        AutoTokenizer.from_pretrained(self.hparams.tokenizer_name, use_fast=True)
        datasets.load_dataset("squad")
        datasets.load_dataset("squad_adversarial", "AddSent")
        datasets.load_dataset("squad_adversarial", "AddOneSent")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before `trainer.fit()` or `trainer.test()`."""

        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name, use_fast=True)

        if not self.collator_fn:
            self.collator_fn = DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=8)

        if not self.dataset:
            self.dataset = datasets.load_dataset("squad")
            self.dataset["test_addsent"] = datasets.load_dataset("squad_adversarial", "AddSent")["validation"]
            self.dataset["test_addonesent"] = datasets.load_dataset("squad_adversarial", "AddOneSent")["validation"]

            # Select training samples if specificed
            if self.hparams.select_train_samples:
                # datasets.Dataset slicing returns a dict
                train_dict = self.dataset["train"].shuffle()[: self.hparams.select_train_samples]
                self.dataset["train"] = datasets.Dataset.from_dict(train_dict)

            # Process
            self.dataset = self.process_data(
                self.dataset,
                self.keep_columns,
                tokenizer=self.tokenizer,
                hparams=self.hparams,
                val_example_id_strings=self.val_example_id_strings,
                test1_example_id_strings=self.test1_example_id_strings,
                test2_example_id_strings=self.test2_example_id_strings,
            )

    def postprocess_func(
        self,
        dataset: Dataset,
        validation_dataset: Dataset,
        original_validation_dataset: Dataset,
        predictions: Dict[int, torch.Tensor],
    ) -> Any:
        return squad_processing.post_processing_function(
            dataset_original=original_validation_dataset,
            examples=original_validation_dataset,
            features=validation_dataset,
            output_dir=self.hparams.output_dir,
            n_best_size=self.hparams.n_best_size,
            predictions=predictions,
            max_answer_length=self.hparams.max_answer_length,
            answer_column_name="answers",
            version_2_with_negative=self.hparams.version_2_with_negative,
            null_score_diff_threshold=self.hparams.null_score_diff_threshold,
        )

    def train_dataloader(self):
        return DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=self.dataset["train"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=True,
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
        addsent_dataloader = DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=self.dataset["test_addsent"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )
        addonesent_dataloader = DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=self.dataset["test_addonesent"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )
        return [addsent_dataloader, addonesent_dataloader]
