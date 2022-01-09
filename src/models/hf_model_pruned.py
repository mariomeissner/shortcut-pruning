import copy
from dataclasses import dataclass
from typing import Any, Dict
import torch

from transformers import AdamW, get_linear_schedule_with_warmup
from nn_pruning.patch_coordinator import (
    SparseTrainingArguments,
    ModelPatchingCoordinator,
)
from src.models.hf_model import SequenceClassificationTransformer

from src.utils.utils import get_logger

log = get_logger(__name__)


class PruningTransformer(SequenceClassificationTransformer):
    def __init__(self, sparse_args: dict, freeze_weights: bool, from_checkpoint: str = None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        if self.hparams.huggingface_model != "bert-base-uncased":
            raise ValueError("Only bert-base-uncased is available for prunning.")

        self.model_patcher = ModelPatchingCoordinator(
            sparse_args=SparseTrainingArguments(**self.hparams.sparse_args),
            device=self.device,
            cache_dir="tmp/",  # Used only for teacher
            model_name_or_path=self.hparams.huggingface_model,
            logit_names="logits",  # TODO
            teacher_constructor=None,  # TODO
        )

        self.model_patcher.patch_model(self.model)

        if self.hparams.freeze_weights:
            self.freeze_non_mask()

    def freeze_non_mask(self):
        for name, param in self.model.named_parameters():
            if name.split(".")[-1] != "mask_scores":
                param.requires_grad = False

    def forward(self, batch: Dict[str, torch.tensor]):
        # Overridden to call scheduler
        self.model_patcher.schedule_threshold(
            step=self.global_step,
            total_step=self.total_training_steps(),
            training=self.training,
        )
        return super().forward(batch)

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):

        outputs = super().training_step(batch, batch_idx)

        prune_reg_loss, _, _ = self.model_patcher.regularization_loss(self.model)

        self.log("train/loss/prune/regularize", prune_reg_loss, prog_bar=False, on_epoch=True, sync_dist=True)

        for stat, val in self.model_patcher.log().items():
            self.log(f"pruning/{stat}", val, prog_bar=False, on_epoch=False, sync_dist=True)

        outputs["loss"] += prune_reg_loss
        return outputs

    def compile_model(self):
        """Returns compiled copy of a debiaed model (NOT in place)."""
        model = copy.deepcopy(self.model)
        removed, heads = self.model_patcher.compile_model(model)

        log.info(f"Compiled model. Removed {removed} / {heads} heads.")

        return model

    def configure_optimizers(self):
        training_args = MockTrainingArgs(
            learning_rate=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        optim_groups = self.model_patcher.create_optimizer_groups(
            self.model, args=training_args, sparse_args=self.hparams.sparse_args
        )

        optimizer = AdamW(optim_groups)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_training_steps()
        )

        return {"optimizer": optimizer, "scheduler": scheduler}


@dataclass
class MockTrainingArgs:
    """Needed for calling model_patcher.create_optimizer_groups."""

    learning_rate: float
    weight_decay: float
