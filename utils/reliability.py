# utils/reliability.py

import torch
import torch.nn.functional as F


def detection_reliability(det_conf, thr=0.5, tau=0.1):
    """
    Soft replacement for TD (detection threshold)

    det_conf: Tensor of shape (N,) in [0,1]
    thr: original TD
    tau: temperature (controls softness)

    returns: reliability weight in [0,1]
    """
    # sigmoid-based soft thresholding
    weight = torch.sigmoid((det_conf - thr) / tau)
    return weight


def recognition_entropy(log_probs):
    """
    Measures uncertainty of recognition output

    log_probs: (B, T, V)
    returns: entropy (B,)
    """
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
    return entropy.mean(dim=-1)                  # (B,)


def recognition_reliability(log_probs, alpha=1.0):
    """
    Soft replacement for TR (recognition threshold)

    Uses entropy: lower entropy → higher reliability
    """
    ent = recognition_entropy(log_probs)
    weight = torch.exp(-alpha * ent)
    return weight.clamp(0.0, 1.0)


def combine_reliability(det_weight, rec_weight):
    """
    Soft replacement for CC (confidence comparison)

    Combines detection + recognition reliability
    """
    return det_weight * rec_weight
