from pathlib import Path
from src.utils.utils import get_logger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

log = get_logger(__name__)


class ModelCheckpointWithCompiling(ModelCheckpoint):
    def __init__(self, compile=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compile = compile

    def on_train_end(self, trainer, pl_module):

        # Save Lightning checkpoint (everything)
        super().on_train_end(trainer, pl_module)

        if self.compile:
            # Save compiled model in HF format
            model = pl_module.compile_model()
            path = Path(self.dirpath) / "model_compiled.bin"
            model.save_pretrained(path)
