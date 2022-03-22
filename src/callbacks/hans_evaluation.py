import pytorch_lightning as pl
from pathlib import Path
from src.datamodules.hf_datamodule import HFDataModule
from src.utils.utils import get_logger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torchmetrics.classification.accuracy import Accuracy

log = get_logger(__name__)


class HansEvaluation(ModelCheckpoint):
    def __init__(self, split_level: str, tokenizer_name: str, enabled=True, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.enabled = enabled
        if not enabled:
            return
            
        if not split_level in ["global", "label", "lexical_types"]:
            raise ValueError("split_level must be one of 'global', 'label', 'lexical_types'.")

        if split_level != "global":
            raise NotImplementedError

        self.split_level = split_level
        self.hans_datamodule = HFDataModule(
            dataset_name="hans",
            subdataset_name=None,
            num_labels=2,
            tokenizer_name=tokenizer_name,
            sentence_1_name="premise",
            sentence_2_name="hypothesis",
        )
        log.info("Setting up HANS callback.")
        self.hans_datamodule.setup()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):

        if not self.enabled:
            return

        # TODO: Split by label when requested
        log.info("Validating on HANS...")
        hans_val_dataloader = self.hans_datamodule.val_dataloader()
        # labels = self.hans_datamodule.dataset["validation"]["labels"]
        device = pl_module.device
        accuracy = Accuracy().to(device)
        for batch in hans_val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            preds = pl_module(batch)[1]  # Get preds
            preds[preds == 2] = 1  # neutral + contradiction = non-entail
            accuracy(preds, labels)
        pl_module.log("hans/val", accuracy.compute(), on_epoch=True)
        log.info("HANS validation done!")

    # @rank_zero_only
    # def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     # TODO: Split by label when requested
    #     log.info("Validating on HANS...")
    #     hans_val_dataloader = self.hans_datamodule.val_dataloader()
    #     # labels = self.hans_datamodule.dataset["validation"]["labels"]
    #     device = pl_module.device
    #     accuracy = Accuracy().to(device)
    #     for batch in tqdm(hans_val_dataloader):
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         labels = batch["labels"]
    #         preds = pl_module(batch)[1]  # Get preds
    #         preds[preds == 2] = 1  # neutral + contradiction = non-entail
    #         accuracy(preds, labels)
    #     pl_module.log("hans/val", accuracy.compute(), on_epoch=True)
    #     log.info("HANS validation done!")
