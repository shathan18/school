"""Loss terms for the joint two-wall optimisation."""
from __future__ import annotations

import torch


def reconstruction(pred, targets):
    """Sum of MSE over both walls. pred/targets: {'A':..,'B':..} tensors [H,W]."""
    return sum(((pred[k] - targets[k]) ** 2).mean() for k in pred)


def sparsity(opacities):
    """Encourage 'floating pieces' -> less material. Mean opacity (L1-ish)."""
    return opacities.clamp(0, 1).mean()


def total_variation(opacities):
    """Anisotropic TV per panel -> smoother regions, fewer slivers before halftone."""
    dv = (opacities[:, 1:, :] - opacities[:, :-1, :]).abs().mean()
    du = (opacities[:, :, 1:] - opacities[:, :, :-1]).abs().mean()
    return dv + du
