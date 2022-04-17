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
from src.datamodules.hf_datamodule import HFDataModule
from src.utils import utils
from bias_utils import load_bias, load_bias_probs

log = utils.get_logger(__name__)


class HFBiasDataModule(HFDataModule):
    def __init__(self, bias_path: str, **kwargs):

        super().__init__(**kwargs)
        self.bias_path = Path(bias_path)
        self.keep_columns += ["bias", "bias_probs"]
        self.save_hyperparameters(logger=False)

    @staticmethod
    def append_bias(example, bias_dict, empty=False):
        if empty:
            bias = [0, 0, 0]
        else:
            bias = bias_dict[str(example["idx"])]
        example["bias_probs"] = bias
        return example

    @staticmethod
    def process_data(dataset: DatasetDict, keep_columns, hparams, tokenizer):
        log.info("Processing dataset!")
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
        # dataset = dataset.filter(lambda sample: sample["labels"] != -1)

        # Append bias!
        log.info(f"Appending bias from file {hparams.bias_path}.")
        bias_dict = load_bias_probs(hparams.bias_path)

        dataset["train"] = dataset["train"].map(
            HFBiasDataModule.append_bias, fn_kwargs={"bias_dict": bias_dict, "empty": False}
        )
        dataset["validation"] = dataset["validation"].map(
            HFBiasDataModule.append_bias, fn_kwargs={"bias_dict": bias_dict, "empty": True}
        )
        dataset["test"] = dataset["test"].map(
            HFBiasDataModule.append_bias, fn_kwargs={"bias_dict": bias_dict, "empty": True}
        )

        # Remove all unnecessary columns
        keep_columns = [column for column in keep_columns if column in dataset["train"].column_names]

        # Set torch format
        dataset.set_format("torch", columns=keep_columns)
        return dataset
