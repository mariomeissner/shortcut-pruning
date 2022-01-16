from pathlib import Path
from src.datamodules.hf_datamodule import HFDataModule
from src.utils.utils import get_logger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

log = get_logger(__name__)


class HansEvaluation(ModelCheckpoint):
    def __init__(self, split_level: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not split_level in ["global", "label", "lexical_types"]:
            raise ValueError("split_level must be one of 'global', 'label', 'lexical_types'.")

        if split_level == "lexical_types":
            raise NotImplementedError

        self.split_level = split_level

        hans_datamodule = HFDataModule(dataset_name="hans")

    def on_validation_end(self, trainer, pl_module):
        pass
        
