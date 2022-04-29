import copy
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from nn_pruning.patch_coordinator import ModelPatchingCoordinator, SparseTrainingArguments
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from transformers import AdamW, AutoModel, get_linear_schedule_with_warmup

from src.utils import utils
from src.losses import GeneralizedCELoss, ProductOfExperts, ReweightByTeacher

log = utils.get_logger(__name__)


class SequenceClassificationTransformer(LightningModule):
    """
    Transformer Model for Sequence Classification.
    Adapted to work with debiasing setups.
    """

    def __init__(
        self,
        huggingface_model: str,
        num_labels: int,
        use_bias_probs: bool,
        loss_fn: str = "crossentropy",
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        generalized_loss_q = 0.7,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        # import ipdb; ipdb.set_trace()
        self.save_hyperparameters(logger=False)

        # Load model and add classification head
        self.model = AutoModel.from_pretrained(huggingface_model)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

        # Init classifier weights according to initialization rules of model
        self.model._init_weights(self.classifier)

        # Apply dropout rate of model
        dropout_prob = self.model.config.hidden_dropout_prob
        log.info(f"Dropout probability of classifier set to {dropout_prob}.")
        self.dropout = nn.Dropout(dropout_prob)

        # loss function (assuming single-label multi-class classification)
        if loss_fn == "crossentropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_fn == "generalized-crossentropy":
            self.loss_fn = GeneralizedCELoss(q=self.hparams.generalized_loss_q)
        elif loss_fn == "reweight-by-teacher":
            self.loss_fn = ReweightByTeacher()
        elif loss_fn == "product-of-experts":
            self.loss_fn = ProductOfExperts()
        else:
            raise ValueError("Unrecognized loss function name.")

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_accs = nn.ModuleList([Accuracy() for _ in range(5)])  # Arbitrarily assume that num_tests <= 5

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self._total_training_steps = None

        # Collect the forward signature
        params = inspect.signature(self.model.forward).parameters.values()
        params = [param.name for param in params if param.kind == param.POSITIONAL_OR_KEYWORD]
        self.forward_signature = params

    def forward(self, batch: Dict[str, torch.tensor]):
        filtered_batch = {key: batch[key] for key in batch.keys() if key in self.forward_signature}
        outputs = self.model(**filtered_batch, return_dict=True)
        pooler = outputs.pooler_output
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)
        preds = torch.argmax(logits, dim=1).int()
        return logits, preds

    def step(self, batch: Dict[str, torch.tensor]):
        logits, preds = self(batch)
        logits = logits.view(-1, self.hparams.num_labels)
        labels = batch["labels"].view(-1)
        if self.hparams.use_bias_probs:
            bias_probs = batch["bias_probs"]
            loss = self.loss_fn(logits, bias_probs, labels)
        else:
            loss = self.loss_fn(logits, labels)
        return loss, preds

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        loss, preds = self.step(batch)
        # log train metrics
        acc = self.train_acc(preds, batch["labels"])
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=False, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": batch["labels"]}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        logits, preds = self(batch)
        # log val metrics
        acc = self.val_acc(preds, batch["labels"])
        # No loss for validation or test because of missing bias_probs!!
        # self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"preds": preds, "targets": batch["labels"]}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def on_test_start(self) -> None:
        self.test_names = self.trainer.datamodule.get_test_names()

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int, dataloader_idx: int = 0):
        logits, preds = self(batch)

        # Nasty HANS hack here (I know I know, but I'm too lazy to find a better solution)
        if self.test_names[dataloader_idx] == "hans":
            preds[preds == 2] = 1  # Merge neutral and contradiction into 'non-entail'

        self.test_accs[dataloader_idx].update(preds, batch["labels"])

        return {"preds": preds, "targets": batch["labels"]}

    def test_epoch_end(self, outputs: List[Any]):
        for test_name, metric in zip(self.test_names, self.test_accs):
            self.log(f"test/{test_name}/acc", metric.compute(), on_step=False, on_epoch=True)

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_acc.reset()
        self.val_acc.reset()

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

        total_steps = self.total_training_steps()
        # Check if warump is a number or a ratio
        if self.hparams.warmup_steps == int(self.hparams.warmup_steps):
            warmup_steps = self.hparams.warmup_steps # number
        else:
            warmup_steps = total_steps * self.hparams.warmup_steps # ratio

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
