"""2D previews: target vs predicted darkness on each wall, plus cross-talk heatmaps."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _show(ax, img, title, cmap="gray_r", vmin=0, vmax=1):
    ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def save_wall_comparison(targets, pred, out_path, crosstalk=None):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for r, w in enumerate(("A", "B")):
        _show(axes[r, 0], targets[w], f"Wall {w}: target")
        _show(axes[r, 1], pred[w], f"Wall {w}: predicted")
        err = abs(pred[w] - targets[w])
        _show(axes[r, 2], err, f"Wall {w}: |error|", cmap="magma", vmax=1)
        ct = crosstalk[w] if crosstalk else err * 0
        _show(axes[r, 3], ct, f"Wall {w}: cross-talk ghost", cmap="magma", vmax=1)
    fig.suptitle("ShadowArt — predicted projection", fontsize=12)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_color_comparison(targets, pred, out_path):
    """targets/pred: {'A','B'} RGB [H,W,3] in [0,1]. Source vs predicted colour projection."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for r, w in enumerate(("A", "B")):
        axes[r, 0].imshow(np.clip(targets[w], 0, 1), origin="lower", aspect="auto")
        axes[r, 0].set_title(f"Wall {w}: source (fitted)", fontsize=9)
        axes[r, 1].imshow(np.clip(pred[w], 0, 1), origin="lower", aspect="auto")
        axes[r, 1].set_title(f"Wall {w}: predicted colour projection", fontsize=9)
        for c in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle("ShadowArt — CMYK colour projection", fontsize=12)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
