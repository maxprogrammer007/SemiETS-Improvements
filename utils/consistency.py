# utils/consistency.py

import torch
import torch.nn.functional as F


def kl_consistency_loss(log_probs_1, log_probs_2):
    """
    KL divergence between two recognition outputs

    log_probs_*: (B, T, V)
    """
    p1 = log_probs_1.exp()
    loss = F.kl_div(log_probs_2, p1, reduction="batchmean", log_target=False)
    return loss


def multi_view_consistency(log_probs_list):
    """
    Enforces consistency across multiple views

    log_probs_list: list of (B, T, V)
    """
    loss = 0.0
    count = 0

    for i in range(len(log_probs_list)):
        for j in range(i + 1, len(log_probs_list)):
            loss += kl_consistency_loss(
                log_probs_list[i],
                log_probs_list[j]
            )
            count += 1

    return loss / max(count, 1)
