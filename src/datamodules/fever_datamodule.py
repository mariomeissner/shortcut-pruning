from pathlib import Path
import datasets
from src.datamodules.hf_datamodule import HFDataModule
from datasets.dataset_dict import DatasetDict
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
REV_LABEL_MAP = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


class FeverDataModule(HFDataModule):
    def __init__(self, num_labels, sentence_1_name, sentence_2_name, data_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def download_dataset(self):
        fever_train = datasets.load_dataset("json", data_files=self.hparams.data_path + "fever_train.jsonl")["train"]
        fever_val = datasets.load_dataset("json", data_files=self.hparams.data_path + "fever_dev.jsonl")["train"]
        fever_symm = datasets.load_dataset("json", data_files=self.hparams.data_path + "fever_symmetric_full.jsonl")[
            "train"
        ]

    def load_dataset(self):
        fever_train = datasets.load_dataset("json", data_files=self.hparams.data_path + "fever_train.jsonl")["train"]
        fever_val = datasets.load_dataset("json", data_files=self.hparams.data_path + "fever_dev.jsonl")["train"]
        fever_symm = datasets.load_dataset("json", data_files=self.hparams.data_path + "fever_symmetric_full.jsonl")[
            "train"
        ]

        fever_train = (
            fever_train.map(lambda sample: {"label": LABEL_MAP[sample["gold_label"]]})
            .remove_columns(["weight"])
            .rename_column("id", "idx")
        )
        fever_val = fever_val.map(lambda sample: {"label": LABEL_MAP[sample["gold_label"]]})
        fever_symm = fever_symm.map(lambda sample: {"label": LABEL_MAP[sample["label"]]}).rename_column(
            "evidence_sentence", "evidence"
        )

        dataset = datasets.DatasetDict(
            {
                "train": fever_train,
                "validation": fever_val,
                "test_symm": fever_symm,
            }
        )
        return dataset

    def test_dataloader(self):

        symm_dataloader = DataLoader(
            multiprocessing_context=self.multiprocessing_context,
            dataset=self.dataset["test_symm"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )
        return [symm_dataloader]

    def get_test_names(self) -> "list[str]":
        return ["symm"]
