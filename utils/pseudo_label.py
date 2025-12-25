import torch


def perturb_images(images, noise_std=0.02):
    """
    Simple pixel-level perturbation
    (acts as proxy for localization noise)
    """
    noise = torch.randn_like(images) * noise_std
    return (images + noise).clamp(0, 1)
