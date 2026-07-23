"""Source flattening: turn a photo / oil painting into bold FLAT colour regions before the shard
decomposition sees it.

Why this exists (see also decompose._importance_map): the overlap decomposer picks each shard's
colour by voting (`color.dominant_rgb`) over a raster patch and concentrates small shards where the
luma GRADIENT is high. On a JPEG or an oil painting, brushstroke / compression texture is high-
frequency EVERYWHERE, so the gradient fires across the whole canvas and per-shard colour votes land
on noisy patches -> the reconstruction comes out blotchy and the shard budget is spent on texture
instead of on the shapes that carry recognition.

Flattening to a few hue-preserving flat colours removes that texture up front: shard boundaries then
land on real colour edges and each shard has one unambiguous colour to vote for. This is the same
approach validated in out_thickness_test/flatten_inputs.py (k-means in RGB, NOT a luma-band
posterise -- so different HUES at the same brightness stay separate instead of merging to mud),
promoted here into a small reusable, seed-deterministic helper the pipeline can call.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

_LUMA = np.array([0.299, 0.587, 0.114])


def boost(rgb, sat=1.15, contrast=1.08, smooth=0.0):
    """GENTLE saturation + contrast (kept mild to avoid an over-processed look), with optional
    Gaussian pre-smoothing (`smooth` = sigma in px) to melt brushwork / pointillist speckle into
    flat regions before quantising. `sat=contrast=1, smooth=0` is a no-op."""
    im = np.asarray(rgb, np.float32)
    if smooth and smooth > 0:
        im = np.stack([ndimage.gaussian_filter(im[..., c], smooth) for c in range(3)], axis=-1)
    g = im @ _LUMA
    im = g[..., None] + sat * (im - g[..., None])            # saturation about luma
    im = 0.5 + contrast * (im - 0.5)                          # contrast about mid-grey
    return np.clip(im, 0.0, 1.0)


def kmeans_quantise(rgb, k=8, iters=12, mask=None, seed=0):
    """Flat colour regions via k-means in RGB (hue-preserving, unlike luma bands). If `mask` is
    given, only those pixels are clustered/recoloured; the rest are left exactly as-is (so a white
    margin is never pulled into the palette). Deterministic for a given `seed`."""
    rgb = np.asarray(rgb, np.float32)
    H, W, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    sel = np.ones(H * W, bool) if mask is None else np.asarray(mask, bool).reshape(-1)
    pts = flat[sel]
    if len(pts) < k:
        return rgb.copy()
    rng = np.random.default_rng(seed)
    cen = pts[rng.choice(len(pts), k, replace=False)].astype(np.float32)
    lab = np.zeros(len(pts), int)
    for _ in range(iters):
        d = ((pts[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                cen[j] = pts[lab == j].mean(0)
    out = flat.copy()
    out[sel] = cen[lab]
    return out.reshape(H, W, 3)


def preprocess_source(rgb, k=8, smooth=1.0, sat=1.15, contrast=1.08, mask=None, seed=0):
    """Flatten `rgb` [H,W,3] in [0,1] to `k` bold, hue-preserving flat colours: gentle
    saturation/contrast + optional Gaussian de-texture (`smooth`), then RGB k-means (`k` clusters).

    `mask` (optional bool [H,W]): restrict clustering to those pixels (e.g. the subject) and leave
    the rest byte-identical to the input -- keeps a white background pure white instead of spending
    a cluster on it. `k=None`/`k<=0` skips quantisation (boost/smooth only)."""
    rgb = np.asarray(rgb, np.float32)
    proc = boost(rgb, sat=sat, contrast=contrast, smooth=smooth)
    if k and k > 0:
        proc = kmeans_quantise(proc, k=k, mask=mask, seed=seed)
    if mask is not None:                                     # restore everything outside the subject
        out = rgb.copy()
        m = np.asarray(mask, bool)
        out[m] = proc[m]
        return out
    return proc
