"""
Produce a consistent bold-FLAT version of every Mona-Lisa-pair candidate, matching the
flat-illustration Mona/Pearl the user likes. Candidates (self-portrait dropped -- a chalk
sketch has no bold colour to flatten):
    mona_louvre (Wall-A anchor) | prado_mona | st_john | pearl

Key fix over a luma-band posterise: quantise in RGB with k-means so different HUES at the
same brightness stay separate (the Prado's red sleeves + blue landscape no longer merge to
brown mud). Paintings also get a saturation+contrast boost first; St. John is isolated from
its near-black ground onto white.

Outputs examples/<name>_flat.png (800x800) + grid out_thickness_test/mona_pairs/flat_inputs.png.
Run:  py out_thickness_test/flatten_inputs.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage

OUTDIR = "out_thickness_test/mona_pairs"; os.makedirs(OUTDIR, exist_ok=True)
RNG = np.random.default_rng(0)


def boost(im, sat=1.15, contrast=1.08, smooth=0.0):
    """GENTLE saturation+contrast (kept mild to avoid the over-processed look), with optional
    Gaussian pre-smoothing to melt brushwork/pointillist speckle into flat regions before
    quantising."""
    if smooth > 0:
        im = np.stack([ndimage.gaussian_filter(im[..., c], smooth) for c in range(3)], -1)
    g = im @ np.array([0.299, 0.587, 0.114])
    im = g[..., None] + sat * (im - g[..., None])            # saturation about luma
    im = 0.5 + contrast * (im - 0.5)                          # contrast about mid-grey
    return np.clip(im, 0, 1)


def kmeans_quantise(im, k=7, iters=12, mask=None):
    """Flat colour regions via k-means in RGB (hue-preserving, unlike luma bands). If `mask`
    is given, only those pixels are clustered/recoloured; the rest are left as-is."""
    H, W, _ = im.shape
    flat = im.reshape(-1, 3)
    sel = np.ones(H * W, bool) if mask is None else mask.reshape(-1)
    pts = flat[sel]
    if len(pts) < k:
        return im
    cen = pts[RNG.choice(len(pts), k, replace=False)]
    for _ in range(iters):
        d = ((pts[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if (lab == j).any():
                cen[j] = pts[lab == j].mean(0)
    out = flat.copy()
    out[sel] = cen[lab]
    return out.reshape(H, W, 3)


def isolate_on_black(im, luma_thr=0.16):
    """Bright subject on near-black ground -> largest bright blob, holes filled, on white."""
    luma = im @ np.array([0.299, 0.587, 0.114])
    subj = luma > luma_thr
    subj = ndimage.binary_opening(subj, iterations=2)
    subj = ndimage.binary_closing(subj, iterations=4)
    lbl, n = ndimage.label(subj)
    if not n:
        return im, np.ones(im.shape[:2], bool), False
    counts = np.bincount(lbl.ravel()); counts[0] = 0
    keep = lbl == counts.argmax()
    keep = ndimage.binary_fill_holes(keep)
    keep = ndimage.binary_closing(keep, iterations=3)
    out = np.ones_like(im); out[keep] = im[keep]
    return out, keep, (0.08 < keep.mean() < 0.85)


def remove_bg_flat(flat, center_frac=0.36, min_cov=0.06, max_cov=0.97):
    """Remove the background from an already k-means FLATTENED image (solid colour regions).

    These are FULL-FRAME portraits (the figure reaches the frame edges), not compact subjects
    on a plain ground, so a plain border-colour cut eats the figure. Instead: a same-colour
    connected region is background iff it TOUCHES THE BORDER but does NOT reach the image
    CENTRE box -- real background (landscape/sky/teal) rings the edges without crossing the
    middle, while the figure's torso does cross it. Removed regions -> white; the largest
    central blob (holes filled) is the subject. Works on coloured grounds the repo's
    neutral-only remove_background can't touch. Returns (rgb_on_white, mask, trustworthy, cov)."""
    H, W, _ = flat.shape
    q = np.round(flat * 255).astype(np.int64)
    key = q[..., 0] * 65536 + q[..., 1] * 256 + q[..., 2]
    edge = np.zeros((H, W), bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    cy0, cy1 = int((0.5 - center_frac / 2) * H), int((0.5 + center_frac / 2) * H)
    cx0, cx1 = int((0.5 - center_frac / 2) * W), int((0.5 + center_frac / 2) * W)
    center = np.zeros((H, W), bool); center[cy0:cy1, cx0:cx1] = True

    background = np.zeros((H, W), bool)
    for colour in np.unique(key):
        m = key == colour
        lbl, n = ndimage.label(m)
        if not n:
            continue
        touches_edge = set(np.unique(lbl[edge & m]).tolist()); touches_edge.discard(0)
        hits_center = set(np.unique(lbl[center & m]).tolist()); hits_center.discard(0)
        bg_comp = touches_edge - hits_center                 # peripheral ring only
        if bg_comp:
            background |= np.isin(lbl, list(bg_comp))
    subject = ~background
    subject = ndimage.binary_closing(subject, iterations=2)
    subject = ndimage.binary_fill_holes(subject)
    slbl, sn = ndimage.label(subject)
    if sn:
        # keep the component covering the image CENTRE most (the figure), not the globally
        # largest -- a dark sfumato painting (Louvre Mona) can otherwise leave a bright sky
        # band as the "largest" component and drop the figure.
        overlap = np.array([((slbl == c) & center).sum() for c in range(1, sn + 1)])
        subject = slbl == (int(overlap.argmax()) + 1)
        subject = ndimage.binary_fill_holes(subject)
    cov = float(subject.mean())
    ok = min_cov <= cov <= max_cov
    out = np.ones_like(flat)
    if ok:
        out[subject] = flat[subject]
    else:
        out = flat                                           # keep full frame, flag
    return out, subject, ok, cov


def fit_square(im, size=800):
    h, w = im.shape[:2]; s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    r = np.asarray(Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS), float) / 255.0
    canvas = np.ones((size, size, 3), float)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), float) / 255.0


def subject_mask_for(coarse):
    """Subject mask from a coarse-posterised copy: luma-isolate only for a UNIFORMLY dark
    ground (all 4 corners dark, e.g. Vermeer's black), else the centre-protected colour-region
    cut. Judging on the brightest corner avoids misrouting a mixed-ground painting (Louvre
    Mona: bright sky top, dark foreground bottom) to the bright-subject isolator, which would
    keep the sky and drop the dark figure. Returns (mask, ok, cov)."""
    lu = coarse @ np.array([0.299, 0.587, 0.114])
    cH, cW = int(0.12 * coarse.shape[0]), int(0.12 * coarse.shape[1])
    corner_med = [np.median(lu[:cH, :cW]), np.median(lu[:cH, -cW:]),
                  np.median(lu[-cH:, :cW]), np.median(lu[-cH:, -cW:])]
    if max(corner_med) < 0.18:                              # ALL corners dark -> true black ground
        _iso, mask, ok = isolate_on_black(coarse)
        return mask, ok, float(mask.mean())
    _out, mask, ok, cov = remove_bg_flat(coarse)
    return mask, ok, cov


# (label, source, mode, smooth)  gentle flatten -- fix the earlier over-processed look.
# smooth = Gaussian sigma (px) applied before quantise, to melt brushwork/pointillism.
JOBS = [
    ("mona_louvre", "examples/Screenshot 2026-07-16 102515.png", "flat",   0.0),  # anchor (Wall A)
    ("prado_mona",  "examples/prado_mona_src.jpg",               "poster", 1.0),  # Prado twin
    ("pearl",       "examples/Screenshot 2026-07-16 102536.png", "flat",   0.0),  # Pearl girl
    ("vangogh",     "examples/vangogh_src.jpg",                  "poster", 3.0),  # Van Gogh self-portrait
    ("klimt_gold",  "examples/klimt_gold_src.jpg",              "poster", 2.5),  # Klimt Woman in Gold
]

# --orig: use the ORIGINAL OIL paintings (both walls), remove only the non-essential
# background, and leave the colours ALONE -- the pipeline's own CMYK shard quantisation does
# the colour approximation. Mask derived from a throwaway coarse posterise; original colours
# (lightly smoothed to tame craquelure) composited inside it on white.
# NB: only the CURRENT batch is listed -- earlier labels' *_nobg_orig.png are already on disk
# (mona_louvre / prado_mona / pearl / vangogh / klimt_gold / sunflowers / lemons). Re-add a line
# here to regenerate one.
JOBS_ORIG = [
    ("sf_cut",  "examples/sf_cut_src.jpg"),   # Four Cut Sunflowers 1887 (same palette, objects)
    ("sf_v2",   "examples/sf_v2_src.jpg"),    # Vase w/ Fifteen Sunflowers (yellow ground - risky)
    ("irises",  "examples/irises_src.jpg"),   # Irises (deliberate palette clash control)
]

NOBG = "--nobg" in sys.argv                                # remove backgrounds -> subject on white
ORIG = "--orig" in sys.argv                                # original oils, code does colour approx

if ORIG:
    results = []
    for label, src in JOBS_ORIG:
        im = load(src)
        coarse = kmeans_quantise(boost(im, smooth=2.0), k=8)     # segmentation only
        mask, ok, cov = subject_mask_for(coarse)
        soft = np.stack([ndimage.gaussian_filter(im[..., c], 1.0) for c in range(3)], -1)
        out = np.ones_like(im)
        if ok:
            out[mask] = soft[mask]
            note = f"orig oil, bg-removed (subject {cov*100:.0f}%)"
        else:
            out = soft
            note = f"orig oil, bg KEPT (subject {cov*100:.0f}% -- flagged)"
        sq = fit_square(out)
        dst = f"examples/{label}_nobg_orig.png"
        Image.fromarray((sq * 255).astype(np.uint8)).save(dst)
        results.append((label, dst, note)); print(f"  {label:14s} <- {os.path.basename(src):40s} {note}")
    grid = f"{OUTDIR}/flat_inputs_orig.png"
    fig, ax = plt.subplots(1, len(results), figsize=(3 * len(results), 3.4))
    for a, (label, dst, note) in zip(ax, results):
        a.imshow(np.asarray(Image.open(dst))); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{label}\n{note}", fontsize=8)
    plt.suptitle("Original oils, background-removed (colour left to the pipeline) — Mona pair candidates", fontsize=12)
    plt.tight_layout(); plt.savefig(grid, dpi=100, bbox_inches="tight"); plt.close()
    print(f"\nsaved grid -> {grid}")
    sys.exit(0)

results = []
for label, src, mode, smooth in JOBS:
    im = load(src)
    note = ""
    if mode == "flat":
        flat = kmeans_quantise(im, k=8)                     # already flat; just unify tone count
    elif mode == "black":
        iso, keep, ok = isolate_on_black(im)
        flat = kmeans_quantise(boost(iso, smooth=1.0), k=6, mask=keep)
        note = "isolated-on-black" + ("" if ok else " [coverage flag]")
    else:
        flat = kmeans_quantise(boost(im, smooth=smooth), k=8)
        note = f"gentle flat (smooth {smooth})"
    suffix = "_flat"
    if NOBG:
        # dark-ground images (e.g. Pearl on black): the figure's dark parts are indistinguishable
        # from the ground for the colour-region cut, so use the luminance isolator instead.
        # Judge on the 4 CORNERS (true background) -- an edge-row median misfires when the figure
        # itself reaches the bottom edge in dark clothing (e.g. Prado Mona).
        lu = flat @ np.array([0.299, 0.587, 0.114])
        cH, cW = int(0.12 * flat.shape[0]), int(0.12 * flat.shape[1])
        corner_luma = np.median(np.concatenate([lu[:cH, :cW].ravel(), lu[:cH, -cW:].ravel(),
                                                lu[-cH:, :cW].ravel(), lu[-cH:, -cW:].ravel()]))
        if corner_luma < 0.15:
            iso, mask, ok = isolate_on_black(flat)
            flat, cov = iso, float(mask.mean())
            note = f"bg-removed via luma (subject {cov*100:.0f}%)" + ("" if ok else " [flag]")
        else:
            flat, _mask, ok, cov = remove_bg_flat(flat)
            note = f"bg-removed (subject {cov*100:.0f}%)" + ("" if ok else " [COVERAGE FLAG -- kept full frame]")
        suffix = "_nobg_flat"
    sq = fit_square(flat)
    dst = f"examples/{label}{suffix}.png"
    Image.fromarray((sq * 255).astype(np.uint8)).save(dst)
    results.append((label, dst, note)); print(f"  {label:14s} <- {os.path.basename(src):40s} {note}")

grid = f"{OUTDIR}/flat_inputs{'_nobg' if NOBG else ''}.png"
fig, ax = plt.subplots(1, len(results), figsize=(3 * len(results), 3.4))
for a, (label, dst, note) in zip(ax, results):
    a.imshow(np.asarray(Image.open(dst))); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"{label}\n{note}", fontsize=8)
plt.suptitle(f"{'Background-removed' if NOBG else 'Flattened'} inputs — Mona pair candidates "
             f"(Wall-A anchor = mona_louvre)", fontsize=12)
plt.tight_layout(); plt.savefig(grid, dpi=100, bbox_inches="tight"); plt.close()
print(f"\nsaved grid -> {grid}")
