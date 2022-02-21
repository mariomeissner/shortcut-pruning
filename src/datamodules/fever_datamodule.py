import datasets
from src.datamodules.hf_datamodule import HFDataModule

"""
Potential resouces for this.
https://github.com/TalSchuster/pytorch-transformers
https://github.com/simonepri/fever-transformers
"""
class FeverDatamodule(HFDataModule):
    def __init__(self, symmetric_path, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def download_dataset(self):
        fever = datasets.load_dataset("fever", "v1.0")
        fever = datasets.DatasetDict(
            {
                "train": fever["train"],
                "validation": fever["labelled_dev"],
                "test": self.load_feversymmetric(),
            }
        )
        return fever

    def load_feversymmetric(self):
        return datasets.load_dataset("json", data_files=self.hparams.symmetric_path)
