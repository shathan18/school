"""Torch backend helpers (CPU by default; the problem sizes are small)."""
from __future__ import annotations

import numpy as np
import torch

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def to_t(x):
    return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEVICE)


def to_np(x):
    return x.detach().cpu().numpy()
