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


# ---------------------------------------------------------------------------
# silhouette / multi-view metrics
# ---------------------------------------------------------------------------
def mark_iou(pred_rgb, target_rgb, thr=0.5):
    """IoU of the DARK MARK -- the shadow -- against the target's dark mark.

    Returns (iou, pred_fg_frac, target_fg_frac).

    The extra two numbers are not decoration. This project has already been burnt once by
    an IoU that silently scored the BACKGROUND instead of the subject (see
    `logo_opt.mark_iou`): on a mostly-white target, background IoU sits near 0.9 and looks
    like a triumph while the actual figure is garbage. Because background and foreground
    IoU are both "an IoU", the mistake is invisible in a log. Reporting the foreground AREA
    FRACTION next to the score makes it self-checking: a plausible portrait silhouette
    covers roughly 0.2-0.6 of the frame, so a pair of fractions up near 0.9 means the
    polarity flipped, and a pred fraction near 0 means nothing was reconstructed at all.
    Always print all three together.
    """
    p = 1.0 - _luma(pred_rgb)                    # darkness: 1 = fully dark, 0 = white wall
    t = 1.0 - _luma(target_rgb)
    pm, tm = p > thr, t > thr
    union = int((pm | tm).sum())
    iou = float((pm & tm).sum() / union) if union else 0.0
    return iou, float(pm.mean()), float(tm.mean())


def distinctness(pred_rgb, targets, thr=0.5):
    """How much each view says something DIFFERENT from the others.

    A multi-view piece can cheat: cast one blob that is a mediocre match to all three
    targets at once. That scores a respectable mean IoU while being, as an artwork, a
    failure -- turning the piece would reveal the same shape three times. This catches it
    by asking how well view i's OWN cast matches view j's target (j != i):

        distinctness = 1 - mean_{i!=j} IoU(pred_i, target_j) / mean_i IoU(pred_i, target_i)

    1.0 = each view matches only its own target (fully distinct); 0.0 = every view's cast
    fits every target equally well, i.e. the views carry no independent information.
    Report it beside mean IoU -- neither number alone is sufficient.
    """
    names = list(pred_rgb)
    if len(names) < 2:
        return 1.0
    self_scores, cross_scores = [], []
    for i in names:
        for j in names:
            s = mark_iou(pred_rgb[i], targets[j], thr)[0]
            (self_scores if i == j else cross_scores).append(s)
    denom = float(np.mean(self_scores))
    if denom <= 1e-9:
        return 0.0
    return float(np.clip(1.0 - np.mean(cross_scores) / denom, 0.0, 1.0))


def evaluate_multiview(targets, pred_rgb, thr=0.5):
    """Per-view accuracy + silhouette IoU + the cross-view distinctness summary.

    Superset of `evaluate_wall_accuracy`: same per-view keys plus `iou`, `pred_fg_frac`,
    `target_fg_frac`, with a `_summary` entry carrying `mean_iou`, `min_iou` and
    `distinctness`. `min_iou` matters because a turntable piece is only as convincing as
    its WORST stop -- averaging hides one dead view behind two good ones.
    """
    out = evaluate_wall_accuracy(targets, pred_rgb)
    for w in out:
        iou, pf, tf = mark_iou(pred_rgb[w], targets[w], thr)
        out[w].update(iou=iou, pred_fg_frac=pf, target_fg_frac=tf)
    ious = [out[w]["iou"] for w in out]
    out["_summary"] = {
        "mean_iou": float(np.mean(ious)),
        "min_iou": float(np.min(ious)),
        "distinctness": distinctness(pred_rgb, targets, thr),
        "views": len(ious),
    }
    return out


def format_multiview_report(m, title="multi-view reconstruction"):
    s = m["_summary"]
    lines = [f"\n=== {title} ({s['views']} views) ==="]
    for w in sorted(k for k in m if k != "_summary"):
        d = m[w]
        lines.append(
            f"view {w}: IoU {d['iou']:.3f} (fg pred {d['pred_fg_frac']:.2f} / "
            f"target {d['target_fg_frac']:.2f})  |  SSIM {d['ssim']:.3f}  |  "
            f"RMSE {d['rmse']:.4f}  |  edge {d['edge_fidelity']:.3f}")
    lines.append(f"mean IoU {s['mean_iou']:.3f}   MIN IoU {s['min_iou']:.3f} "
                 f"(the worst stop is what the viewer judges the piece by)   "
                 f"distinctness {s['distinctness']:.3f}")
    lines.append("  (fg fractions are a polarity self-check: a portrait silhouette should "
                 "cover ~0.2-0.6 of the frame -- values near 0.9 mean the mark/background "
                 "sense flipped and the IoU is meaningless)")
    return "\n".join(lines)


def shard_view_duty(scene, table, fragments, targets, white_thr=0.90):
    """What fraction of shards actually earn their keep in more than one view.

    Multiplexing is the whole economic argument for a turntable piece: a shard that only
    ever contributes to one stop is dead weight in the other two. For each shard this
    projects its wall-space centroid back to its host panel, then forward to EVERY view,
    and counts the views where it lands on subject content.

    Note this is a geometric duty measure -- it says the shard's shadow falls somewhere
    the image is dark, not that it is the right tone there. Treat it as an upper bound and
    read it beside per-view IoU, never instead of it.

    Returns {n_shards, mean_views, frac_multi (>=2 views), frac_all}.
    """
    from .geometry import homography as H
    from .targets import color as C

    subj = {w: C.subject_mask(targets[w], white_thr) for w in scene.walls}
    counts = []
    for f in fragments:
        home, panel_name = f.get("wall"), f.get("panel")
        if home is None or panel_name not in {p.name for p in scene.panels}:
            continue
        cen = f.get("centroid_m")
        if cen is None:
            continue
        uv = H.apply_homography(table[(panel_name, home)].H_wp, np.asarray(cen, float)[None])
        n = 0
        for w, wall in scene.walls.items():
            ab = H.apply_homography(table[(panel_name, w)].H_pw, uv)[0]
            Hn, Wn = subj[w].shape
            c = int(ab[0] / wall.width * Wn)
            r = int(ab[1] / wall.height * Hn)
            if 0 <= r < Hn and 0 <= c < Wn and subj[w][r, c]:
                n += 1
        counts.append(n)
    if not counts:
        return {"n_shards": 0, "mean_views": 0.0, "frac_multi": 0.0, "frac_all": 0.0}
    a = np.asarray(counts)
    return {"n_shards": int(a.size), "mean_views": float(a.mean()),
            "frac_multi": float((a >= 2).mean()), "frac_all": float((a >= len(scene.walls)).mean())}

