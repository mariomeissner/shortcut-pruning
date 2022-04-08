import copy
import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_metric
from nn_pruning.patch_coordinator import ModelPatchingCoordinator, SparseTrainingArguments
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric, Metric
from torchmetrics.classification.accuracy import Accuracy
from transformers import AdamW, AutoModel, AutoModelForQuestionAnswering, get_linear_schedule_with_warmup

from src.losses import ReweightByTeacher
from src.utils import utils

log = utils.get_logger(__name__)


class QuestionAnsweringTransformer(LightningModule):
    """
    Transformer Model for Question Answering (SQuAD for now).
    """

    def __init__(
        self,
        huggingface_model: str,
        use_teacher_probs: bool,
        loss_fn: str = "crossentropy",
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        batch_size: int = 16,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Load model and add classification head
        self.model = AutoModelForQuestionAnswering.from_pretrained(huggingface_model)

        # loss function (assuming single-label multi-class classification)
        if loss_fn == "crossentropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_fn == "reweight-by-teacher":
            self.loss_fn = ReweightByTeacher()

        self._total_training_steps = None

        # Collect the forward signature
        params = inspect.signature(self.model.forward).parameters.values()
        params = [param.name for param in params if param.kind == param.POSITIONAL_OR_KEYWORD]
        self.forward_signature = params

    def forward(self, batch: Dict[str, torch.tensor]):
        filtered_batch = {key: batch[key] for key in batch.keys() if key in self.forward_signature}
        outputs = self.model(**filtered_batch, return_dict=True)
        return outputs

    def step(self, batch: Dict[str, torch.tensor]):
        """Implement debiasing here later."""
        return self(batch)

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        outputs = self.step(batch)
        self.log("train/loss", outputs.loss, on_step=True, on_epoch=False, prog_bar=False)
        return {"loss": outputs.loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        batch.pop("offset_mapping")
        example_ids = batch.pop("example_id")
        outputs = self(batch)
        self.val_metric.update(example_ids, outputs.start_logits, outputs.end_logits)

    def validation_epoch_end(self, outputs: List[Any]):
        metric_dict = self.val_metric.compute()
        self.log("val/exact_match", metric_dict["exact_match"], on_epoch=True)
        self.log("val/f1", metric_dict["f1"], on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        self.val_metric.reset()

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: int):
        assert dataloader_idx in (0, 1)
        batch.pop("offset_mapping")
        example_ids = batch.pop("example_id")
        outputs = self(batch)
        if dataloader_idx == 0:
            self.test1_metric.update(example_ids, outputs.start_logits, outputs.end_logits)
        else:
            self.test2_metric.update(example_ids, outputs.start_logits, outputs.end_logits)

    def test_epoch_end(self, outputs: List[Any]):
        test1_metric_dict = self.test1_metric.compute()
        test2_metric_dict = self.test2_metric.compute()
        self.log("test/addsent/exact_match", test1_metric_dict["exact_match"], on_epoch=True)
        self.log("test/addsent/f1", test1_metric_dict["f1"], on_epoch=True)
        self.log("test/addonesent/exact_match", test2_metric_dict["exact_match"], on_epoch=True)
        self.log("test/addonesent/f1", test2_metric_dict["f1"], on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.test1_metric.reset()
        self.test2_metric.reset()

    def on_epoch_end(self):
        pass

    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""

        # Special handling for manual evaluation case
        if getattr(self, "trainer", None) is None:
            if not self._total_training_steps:
                log.warn(
                    "Could not compute total_training_steps, returninng -1 instead.\n"
                    "This should only happen if you are evaluating with your own prediction loop.\n"
                    "Make sure to set model.eval() before calling model inference, so that pruning scheduler sets final threshold."
                )
                self._total_training_steps = -1
            return self._total_training_steps

        # Have already computed before
        if self._total_training_steps:
            return self._total_training_steps

        # First call
        else:
            if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
                dataset_size = self.trainer.limit_train_batches
            elif isinstance(self.trainer.limit_train_batches, float):
                # limit_train_batches is a percentage of batches
                dataset_size = len(self.trainer.datamodule.train_dataloader())
                dataset_size = int(dataset_size * self.trainer.limit_train_batches)
            else:
                dataset_size = len(self.trainer.datamodule.train_dataloader())

            num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
            if self.trainer.tpu_cores:
                num_devices = max(num_devices, self.trainer.tpu_cores)

            effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
            max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

            if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
                self._total_training_steps = self.trainer.max_steps
            self._total_training_steps = max_estimated_steps
            return self._total_training_steps

    def setup(self, stage: str):
        """Called at the beginning of train/val/test. Set up metrics here for access to dataset."""
        datamodule = self.trainer.datamodule
        self.val_metric = SquadMetric(
            postprocess_func=partial(
                datamodule.postprocess_func,
                dataset=datamodule.dataset,
                validation_dataset=datamodule.dataset["validation"],
                original_validation_dataset=datamodule.dataset["validation_original"],
            ),
            example_id_strings=datamodule.val_example_id_strings,
        )
        self.test1_metric = SquadMetric(
            postprocess_func=partial(
                datamodule.postprocess_func,
                dataset=datamodule.dataset,
                validation_dataset=datamodule.dataset["test_addsent"],
                original_validation_dataset=datamodule.dataset["test_addsent_original"],
            ),
            example_id_strings=datamodule.test1_example_id_strings,
        )
        self.test2_metric = SquadMetric(
            postprocess_func=partial(
                datamodule.postprocess_func,
                dataset=datamodule.dataset,
                validation_dataset=datamodule.dataset["test_addonesent"],
                original_validation_dataset=datamodule.dataset["test_addonesent_original"],
            ),
            example_id_strings=datamodule.test2_example_id_strings,
        )

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]
        optim_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        print(f"{self.hparams.learning_rate =}")
        optimizer = AdamW(optim_groups, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_training_steps(),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


class SquadMetric(Metric):
    def __init__(self, postprocess_func, example_id_strings):
        super().__init__(compute_on_step=False)
        self.metric = load_metric("squad")
        self.postprocess_func = postprocess_func
        self.example_id_strings = example_id_strings
        self.add_state("start_logits", [])
        self.add_state("end_logits", [])
        self.add_state("example_ids", [])

    def update(self, example_ids: torch.Tensor, start_logits: torch.Tensor, end_logits: torch.Tensor):
        self.example_ids += example_ids
        self.start_logits += start_logits
        self.end_logits += end_logits

    def compute(self):
        reverse_lookup = {i: s for s, i in self.example_id_strings.items()}
        example_ids = [reverse_lookup[i.item()] for i in self.example_ids]
        predictions = (
            torch.stack(self.start_logits).cpu().numpy(),
            torch.stack(self.end_logits).cpu().numpy(),
            example_ids,
        )
        predictions, references = self.postprocess_func(predictions=predictions)
        value = self.metric.compute(predictions=predictions, references=references)
        return value
