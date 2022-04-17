# Reweight using unknown-bias
from torch import nn
from torch.nn import functional as F
import torch


class ReweightByTeacher(nn.Module):
    def forward(self, logits, bias_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction="none")
        one_hot_labels = torch.eye(logits.size(1), device=logits.device)[labels]
        weights = 1 - (one_hot_labels * bias_probs).sum(1)
        return (weights * loss).sum() / weights.sum()


class ProductOfExperts(nn.Module):
    def forward(self, logits, bias_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        log_model_probs = F.log_softmax(logits, dim=1)
        log_bias_probs = torch.log(bias_probs)
        return F.cross_entropy(log_model_probs + log_bias_probs, labels)
