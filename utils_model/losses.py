import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, num_relations=1):
        super().__init__()

        self.temperature = temperature

    def forward(self, rate_embds, pred, ys):
        # rate_embds: [n, 128]
        # pred: [bz, 128]
        # ys: [bz]

        n = len(rate_embds)

        dot_result = pred @ rate_embds.T / self.temperature
        # for numerical stability
        logits_max, _ = torch.max(dot_result, dim=1, keepdim=True)
        logits = dot_result - logits_max
        # [bz, n]

        onehots = F.one_hot(ys, num_classes=n)
        # [bz, n]

        positive = (logits * onehots).sum(-1)
        # [bz]
        total = torch.logsumexp(logits, -1)
        # [bz]

        loss = -torch.mean(positive - total)
        # [bz]

        return loss
