# Reweight using unknown-bias
from torch import nn
from torch.nn import functional as F
import torch


class ReweightByTeacher(nn.Module):
    def forward(self, logits, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction="none")
        one_hot_labels = torch.eye(logits.size(1), device=logits.device)[labels]
        weights = 1 - (one_hot_labels * teacher_probs).sum(1)
        return (weights * loss).sum() / weights.sum()
