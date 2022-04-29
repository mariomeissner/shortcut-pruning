import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from src.datamodules.hf_datamodule import HFDataModule
from src.utils.utils import get_logger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torchmetrics.classification.accuracy import Accuracy

log = get_logger(__name__)


class RecordPreds(ModelCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.counter = 0

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):

        train_dataloader = trainer.datamodule.train_dataloader(shuffle=False)
        val_dataloader = trainer.datamodule.val_dataloader()
        device = pl_module.device
        train_outputs = []
        val_outputs = []

        # Extract train preds
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            train_outputs.append(pl_module(batch)[0])
        train_outputs = torch.cat(train_outputs, dim=0).detach().cpu().numpy()
        output_path = Path("train_predictions/")
        output_path.mkdir(exist_ok=True)
        output_path = output_path / f"{self.counter}"
        np.save(output_path, train_outputs)

        # Extract validation preds
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            val_outputs.append(pl_module(batch)[0])
        val_outputs = torch.cat(val_outputs, dim=0).detach().cpu().numpy()
        output_path = Path("val_predictions/")
        output_path.mkdir(exist_ok=True)
        output_path = output_path / f"{self.counter}"
        np.save(output_path, val_outputs)

        self.counter += 1
