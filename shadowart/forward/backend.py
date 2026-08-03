"""Torch backend helpers.

Device selection: CUDA when available, else CPU. Override with the env var
`SHADOWART_DEVICE` (e.g. `cpu`, `cuda`, `cuda:1`) -- useful to force a CPU run for a
reproducibility comparison, or to pin a specific GPU. The forward renderer builds every
cached tensor on `DEVICE`, so this one switch moves the whole solve.
"""
from __future__ import annotations

import os

import numpy as np
import torch


def _select_device() -> torch.device:
    want = os.environ.get("SHADOWART_DEVICE", "").strip()
    if want:
        dev = torch.device(want)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"SHADOWART_DEVICE={want!r} but this torch build reports no CUDA device. "
                f"Install a CUDA build of torch, or set SHADOWART_DEVICE=cpu.")
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _select_device()
DTYPE = torch.float32


def to_t(x):
    """Array or tensor -> tensor on the active device, as float32.

    Tensors are moved with `.to()` rather than rebuilt, so an opacity field a caller is
    optimising keeps its autograd history (and a tensor already on the right device/dtype
    passes through untouched). Without this, a CPU tensor handed to the renderer meets
    device-resident sampling grids and `grid_sample` fails.
    """
    if torch.is_tensor(x):
        return x.to(device=DEVICE, dtype=DTYPE)
    return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEVICE)


def to_np(x):
    return x.detach().cpu().numpy()
