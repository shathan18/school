"""Load target images into wall-space darkness maps, and make demo targets.

A "target" is a darkness map in [0,1] at the wall resolution: 1 = full shadow (ink),
0 = lit wall. Images are luminance-inverted (dark drawing -> shadow) and flipped so the
top of the picture lands at the top of the wall (wall rasters index z upward).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def load_target(path, wall_res, invert=True):
    """Load an image -> darkness map [rows(z), cols(width)] in [0,1]."""
    Hn, Wn = wall_res
    img = Image.open(path).convert("L").resize((Wn, Hn), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    darkness = (1.0 - a) if invert else a
    return np.flipud(darkness).copy()          # image-top -> wall-top


def normalize_silhouette(in_path, out_path, size=800, content_frac=0.9,
                         align="bottom", pad_frac=0.05, thresh=128):
    """Normalise a silhouette to a fixed SQUARE canvas at a consistent scale.

    Composites over white (handles transparent RGBA), crops to the inked content,
    scales it so its height is `content_frac` of the canvas (preserving aspect, never
    exceeding the width), and places it centred horizontally, aligned to a common
    baseline. Result is a grayscale black-on-white PNG, ideal as a target image.
    Use it to make two differently-sized inputs the same size + scale.
    """
    img = Image.open(in_path).convert("RGBA")
    comp = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img).convert("L")
    darkness = 255 - np.asarray(comp)
    ys, xs = np.where(darkness > thresh)
    if xs.size == 0:
        raise ValueError(f"no dark content found in {in_path}")
    crop = comp.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    cw, ch = crop.size
    new_h = int(round(size * content_frac))
    new_w = max(1, int(round(cw * new_h / ch)))
    max_w = int(round(size * content_frac))
    if new_w > max_w:                                  # too wide -> fit width instead
        new_h = max(1, int(round(new_h * max_w / new_w))); new_w = max_w
    crop = crop.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    x = (size - new_w) // 2
    pad = int(round(size * pad_frac))
    y = {"bottom": size - new_h - pad, "top": pad}.get(align, (size - new_h) // 2)
    canvas.paste(crop, (x, y))
    canvas.save(out_path)
    return str(out_path), (size, size)


def remove_background(in_path, out_path=None, bg_tol=0.13, sat_tol=0.12,
                      close_iter=3, keep_largest=True):
    """Isolate a centered subject on a plain/light background -> subject on WHITE.

    ShadowArt reconstructs a *centered subject with a clear silhouette* far better than a
    full-frame painting (a full frame spreads the shard budget over background the shadow
    can never reproduce, so it collapses to flat colour blobs). This removes a plain
    background so the whole budget lands on the object.

    Method (no ML, numpy/scipy only): sample the four corners for the background colour,
    flood-fill from the image border the connected region that is within `bg_tol` of it AND
    low-saturation (so a coloured subject touching the edge is not eaten), then treat
    everything else as subject. `keep_largest` keeps only the biggest subject blob (drops
    stray specks). Holes inside the subject are filled. Returns (out_path, coverage_frac).

    Works on the common museum/product case (object on white/grey/neutral). For a busy or
    same-colour-as-subject background it will under-segment -- fall back to a manual cut-out
    (e.g. the pre-made *_nobg.png files) in that case."""
    img = Image.open(in_path).convert("RGB")
    arr = np.asarray(img, np.float32) / 255.0
    H, W = arr.shape[:2]
    mx = arr.max(-1); mn = arr.min(-1)
    sat = mx - mn

    # background reference = mean of the four corner patches (robust to a single dark corner)
    c = max(2, int(0.03 * min(H, W)))
    corners = np.concatenate([arr[:c, :c].reshape(-1, 3), arr[:c, -c:].reshape(-1, 3),
                              arr[-c:, :c].reshape(-1, 3), arr[-c:, -c:].reshape(-1, 3)])
    bg_rgb = corners.mean(0)
    near_bg = (np.abs(arr - bg_rgb).max(-1) <= bg_tol) & (sat <= sat_tol)

    # only background that is CONNECTED to the border counts (protects same-colour interior
    # regions of the subject, e.g. white shirt on white background stays subject if enclosed)
    border = np.zeros((H, W), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    lbl, n = ndimage.label(near_bg)
    bg_labels = set(np.unique(lbl[border & near_bg]))
    background = np.isin(lbl, list(bg_labels)) if bg_labels else np.zeros((H, W), bool)
    subject = ~background
    subject = ndimage.binary_closing(subject, iterations=close_iter)
    subject = ndimage.binary_fill_holes(subject)
    if keep_largest:
        slbl, sn = ndimage.label(subject)
        if sn:
            counts = np.bincount(slbl.ravel()); counts[0] = 0
            subject = slbl == counts.argmax()

    coverage = float(subject.mean())
    # Guardrail: corner-flood-fill is only reliable for a plain background framing a compact
    # subject. If the "subject" fills almost the whole frame (background not found -> nothing
    # removed) or almost none of it (subject mistaken for background -> inverted, e.g. a light
    # face on a dark ground), the result is untrustworthy -- keep the ORIGINAL and flag it, so
    # a bad auto-cut never silently poisons the run. Use a manual cutout for those.
    trustworthy = 0.10 <= coverage <= 0.92
    if trustworthy:
        out = np.ones_like(arr)                            # composite subject onto white
        out[subject] = arr[subject]
    else:
        out = arr
    out_path = out_path or (str(Path(in_path).with_suffix("")) + "_nobg.png")
    Image.fromarray((out * 255).astype(np.uint8)).save(out_path)
    return str(out_path), coverage, trustworthy


def save_darkness(darkness, path):
    """Save a darkness map back to a PNG (for previews / debugging)."""
    img = (np.flipud(np.clip(darkness, 0, 1)) * 255).astype(np.uint8)
    Image.fromarray(255 - img).save(path)      # show as dark-on-white


# ---------------------------------------------------------------------------
# Demo target generators (so the pipeline runs with no external inputs).
# ---------------------------------------------------------------------------
def _heart(draw, S):
    cx, cy, r = S * 0.5, S * 0.42, S * 0.26
    draw.ellipse([cx - r, cy - r, cx, cy], fill=0)
    draw.ellipse([cx, cy - r, cx + r, cy], fill=0)
    draw.polygon([(cx - r, cy - r * 0.15), (cx + r, cy - r * 0.15),
                  (cx, cy + r * 1.5)], fill=0)


def _star(draw, S, points=5):
    cx, cy = S * 0.5, S * 0.5
    rO, rI = S * 0.42, S * 0.17
    verts = []
    for i in range(points * 2):
        ang = -math.pi / 2 + i * math.pi / points
        rr = rO if i % 2 == 0 else rI
        verts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    draw.polygon(verts, fill=0)


def make_sample_images(out_dir, size=512):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fn in (("a_heart", _heart), ("b_star", _star)):
        img = Image.new("L", (size, size), 255)
        fn(ImageDraw.Draw(img), size)
        p = out_dir / f"{name}.png"
        img.save(p)
        paths[name] = str(p)
    return paths
