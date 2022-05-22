from pathlib import Path
from typing import Any, List, Tuple
from src.utils.utils import get_logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.types import _METRIC, _PATH, STEP_OUTPUT

log = get_logger(__name__)


class ModelCheckpointWithSchedule(ModelCheckpoint):
    def __init__(self, triggers: List[int], intervals: List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(triggers) == len(intervals)
        self.triggers = triggers
        self.intervals = intervals
        self.schedule_index = 0
        log.info(f"Initialized ModelCheckpointWithSchedule with initial interval: {self._every_n_train_steps}.")

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        step = trainer.global_step
        if self.schedule_index < len(self.triggers) and step >= self.triggers[self.schedule_index]:
            next_rate = self.intervals[self.schedule_index]
            log.info(f"Checkpoint scheduler triggered at step: {step}. New interval: {next_rate}.")
            self._every_n_train_steps = next_rate
            self.schedule_index += 1
