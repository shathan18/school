"""Quantify how faithfully the rendered wall shadow reconstructs the source image.

Plain colour-error metrics (MSE/PSNR) can look deceptively fine on a blurred
reconstruction -- averaging still lands near the right hue over a region even when all
the fine structure is gone. Blur specifically destroys local contrast/structure and
high-frequency edges, so SSIM and edge-fidelity are included to catch exactly that
failure mode (e.g. the "blurry spots of colour" a too-aggressive shard scatter produces),
separately from whether the colours themselves are correct.

No extra dependency: everything here is numpy + scipy.ndimage (already a project
dependency), no scikit-image required.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def mse(pred, target):
    return float(np.mean((np.asarray(pred, np.float64) - np.asarray(target, np.float64)) ** 2))


def psnr(pred, target, data_range=1.0):
    m = mse(pred, target)
    if m <= 1e-12:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / m)


def _luma(rgb):
    """Perceptual grayscale (Rec. 601) from an [...,3] RGB array in [0,1]."""
    rgb = np.asarray(rgb, np.float64)
    return rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140


def ssim(pred, target, sigma=1.5, data_range=1.0):
    """Mean structural similarity index (Wang et al. 2004), grayscale, Gaussian-windowed.

    1.0 = identical structure; drops sharply (independent of colour-accuracy metrics) when
    local contrast/detail is smoothed away, which is exactly what shard-position jitter that
    exceeds the render's resolvable detail does to the projected image."""
    p, t = _luma(pred), _luma(target)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    gf = lambda x: ndimage.gaussian_filter(x, sigma)
    mu_p, mu_t = gf(p), gf(t)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    var_p = gf(p * p) - mu_p2
    var_t = gf(t * t) - mu_t2
    cov_pt = gf(p * t) - mu_pt
    ssim_map = ((2 * mu_pt + C1) * (2 * cov_pt + C2)) / ((mu_p2 + mu_t2 + C1) * (var_p + var_t + C2))
    return float(ssim_map.mean())


def edge_fidelity(pred, target):
    """Normalised cross-correlation between gradient-magnitude maps of pred vs. target.

    Isolates whether recognisable edges/detail survived, independent of overall colour
    accuracy -- a wash of the right average colour but no structure scores near 0 here even
    if its MSE/PSNR look fine."""
    p, t = _luma(pred), _luma(target)
    gp = np.hypot(ndimage.sobel(p, axis=0), ndimage.sobel(p, axis=1))
    gt = np.hypot(ndimage.sobel(t, axis=0), ndimage.sobel(t, axis=1))
    gp, gt = gp - gp.mean(), gt - gt.mean()
    denom = np.sqrt((gp ** 2).sum() * (gt ** 2).sum())
    return float((gp * gt).sum() / denom) if denom > 1e-9 else 0.0


def evaluate_wall_accuracy(targets, pred_rgb):
    """{'A':..., 'B':...} -> per-wall dict of {mse, rmse, psnr_db, ssim, edge_fidelity}.

    `targets`/`pred_rgb`: {'A','B'} RGB arrays [H,W,3] in [0,1] (same shape/orientation the
    renderer and preview already use -- e.g. `C.load_color_target` output vs.
    `Renderer.render_color_np` output)."""
    out = {}
    for w in targets:
        t = np.clip(targets[w], 0, 1)
        p = np.clip(pred_rgb[w], 0, 1)
        m = mse(p, t)
        out[w] = {
            "mse": m,
            "rmse": float(np.sqrt(m)),
            "psnr_db": psnr(p, t),
            "ssim": ssim(p, t),
            "edge_fidelity": edge_fidelity(p, t),
        }
    return out


def format_accuracy_report(metrics, title="wall reconstruction accuracy"):
    lines = [f"\n=== {title} ==="]
    for w, m in sorted(metrics.items()):
        lines.append(
            f"Wall {w}: RMSE {m['rmse']:.4f}  |  PSNR {m['psnr_db']:.1f} dB  |  "
            f"SSIM {m['ssim']:.3f}  |  edge-fidelity {m['edge_fidelity']:.3f}"
        )
    lines.append("  (RMSE/PSNR = colour accuracy; SSIM/edge-fidelity = structural/detail "
                 "accuracy -- low SSIM or edge-fidelity with a fine RMSE means the image is "
                 "washing out into colour blobs even though the average colour is close)")
    return "\n".join(lines)
