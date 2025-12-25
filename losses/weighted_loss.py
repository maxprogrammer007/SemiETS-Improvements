# losses/weighted_loss.py

import torch
import torch.nn as nn


class WeightedCTCLoss(nn.Module):
    def __init__(self, blank=0, reduction="mean"):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)
        self.reduction = reduction

    def forward(
        self,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        weights=None
    ):
        """
        log_probs: (T, B, V)
        targets:  (sum(target_lengths))
        input_lengths: (B,)
        target_lengths: (B,)
        weights: (B,) reliability weights in [0,1]
        """

        loss = self.ctc(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )  # (B,)

        if weights is not None:
            loss = loss * weights.detach()  # do not backprop through weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
