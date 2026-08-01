"""Face pre-test: do STARK TWO-TONE graphic faces survive a 300-shard budget where
smooth oil-painting faces do not?

!! SUPERSEDED -- READ THIS FIRST. The real solver run (`face_render300.py`) DISPROVED
!! this script's central methodological claim. Kept because the correction is the useful
!! part, but do NOT use its verdicts to screen images.
!!
!! What this script claimed: that simulating 150 shards/wall as an ideal uniform 12x12
!! grid is an OPTIMISTIC UPPER BOUND, so anything failing here fails for certain.
!! What actually happened: the real 300-shard render of Poe and Dostoevsky is CLEANLY
!! RECOGNISABLE (face-box detail retention 0.745, and it lands at ~208 shards, well
!! under the 300 budget), while this script scored the two-tone group BELOW the smooth
!! oil controls. The grid proxy is a LOWER bound for flat-mass art, not an upper one.
!! Why it was wrong -- the grid gets three things the solver does not have to suffer:
!!   1. it spends cells on clear-white regions. The solver spends ZERO shards there
!!      (white = no shard), so a two-tone poster's lit face is free and the whole budget
!!      lands on the dark masses. Measured: `ink_area` 0.44 for Poe/Dostoevsky vs 0.84
!!      for the oils -- the two-tone pair gets ~2x the effective resolution per shard.
!!   2. its cell boundaries are on a fixed lattice. Real shards are Voronoi regions
!!      clipped to the subject mask, so shard edges FOLLOW the ink boundary -- which for
!!      a two-tone image is exactly where all the information is.
!!   3. it ignores `detail_bias`, which concentrates shards on transitions.
!! Generalisation: a uniform-grid resolution argument systematically under-rates flat,
!! hard-edged art and is roughly fair only for smooth, mid-tone-everywhere sources.

BACKGROUND (corrections_note.md sec.2): photographic / oil-painted faces are
RESOLUTION-limited, not colour-limited. They only resolve recognisable features at
~2750 shards, ~9x our 300-shard budget. Open question this script tests: a
high-contrast, near-posterised graphic face (bold black/white masses, no smooth
modelling) might carry identity in the COARSE band only -- in which case it could
survive 300 shards where a photographic face cannot. (Answer, from the real render:
yes, if the source is flat MASS. Hatched line-work still fails -- see Wagner.)

WHY blur_pretest.py's SCORE IS THE WRONG TOOL HERE
--------------------------------------------------
`blur_pretest.py` scores "one dominant bold shape on a contrasting ground". A head is a
bold compact shape, so ANY portrait -- recognisable or not -- scores well. It would
happily pass a featureless head-shaped blob. That metric answers "is there a bold
silhouette?", not "is this still that specific person?". We do not reuse it.

WHAT WE MEASURE INSTEAD
-----------------------
1. SHARD-SCALE SIMULATION (physical proxy, not an arbitrary blur).
   300 shards / 2 walls = ~150 per wall. `targets.color.load_color_target` fits the
   cropped subject to ~92% of the wall, so those 150 patches land on the head. An ideal
   uniform tiling of 150 patches is a sqrt(150) ~= 12 x 12 grid, i.e. **one shard
   subtends ~8.3% of head width**. We simulate by area-averaging the head-tight crop to
   12x12 and snapping each cell to the buildable `noir` palette (white + 3 greys + K).
   THIS WAS CLAIMED TO BE AN OPTIMISTIC UPPER BOUND. IT IS NOT -- see the banner above;
   the real render beats it comfortably on flat-mass sources. Treat the grid sim as a
   rough intuition pump for smooth sources only.

2. FACE-BOX DETAIL RETENTION (the fix for the obvious trap).
   A first pass measured band energy over the whole head crop and ranked the *Mona Lisa*
   top -- because at "feature scale" the dominant energy is the long sharp EDGE of the
   hair/dress mass, not the eyes, and a coarse grid reproduces a long edge well. So we
   restrict every detail metric to a hand-set FACE BOX (brow-to-chin, cheek-to-cheek),
   which excludes the silhouette edge. Retention is a PROJECTION,
   <detail_sim, detail_orig> / <detail_orig, detail_orig>, so the fresh high-frequency
   energy injected by cell boundaries (uncorrelated with the source) scores ~0 instead of
   being rewarded as if it were signal.

3. FEATURE-BAND READOUT -- literally "do the eyes/nose/mouth land?".
   Row-mean luminance profile down the face box. A face that still reads has a dark
   brow/eye band in the upper half and a dark mouth/moustache band in the lower half. We
   detect local minima and report their prominence as a fraction of the face box's tonal
   range. `n_bands` = how many survive, `band_depth` = how deep the strongest is.

4. GRID-PHASE RANK-1 IDENTITY.
   A first pass scored 10/10 and was worthless -- the probe was the shard-sim of the
   exact same crop, so it trivially matched itself (same trap as
   `panel_search.joint_intersection_pct` reading 100% for everything, per
   shadowart-noise.md: never rank on a metric that passes everything). Fixed: the probe
   is simulated with the shard lattice SHIFTED BY HALF A CELL and rescaled ~6%, because
   in a real build the shard lattice does not politely align with someone's eyes. If
   identity survives only for one lucky grid phase, it is not a property of the image.

5. THE THUMBNAILS ARE THE REAL EVIDENCE. 1-4 are proxies. The contact sheet answers the
   actual question, and rank-1 separability among 9 blobs is far easier than a human
   naming one, so a pass there means "not yet ruled out", never "recognisable".

CANDIDATE POOL
--------------
TEST: Felix Vallotton (d.1925, public domain) woodcut portraits -- the canonical stark
two-tone graphic face. Secular writers/composers only (respects the project's
no-religion / no-politics rule). Two of them (Wagner, Ibsen) are high-contrast but
LINE-HATCHED rather than flat-mass, kept on purpose to separate "high contrast" from
"bold mass" -- they are not the same property.
CONTROL: the smooth oil-painting faces in examples/ that the ~2750-shard finding came
from, run through the IDENTICAL pipeline so the comparison is in-sample, not quoted.

DROPPED: Vallotton's 1891 self-portrait woodcut -- it is a full figure in a landscape,
not a portrait head, so it is not the category under test.

CAVEAT RECORDED ON PURPOSE: this direction is GRAYSCALE. Grayscale discards hue, so the
colour-compatibility / colour-agreeing-double-duty result (shadowart-noise.md,
report_team.md) DOES NOT APPLY to any pair drawn from this pool.

Outputs -> out_faces_hc/
  pretest_contact.png    head crop | 2-tone poster | 300-shard sim | profile  (evidence)
  pretest_scores.json    metrics per candidate
  pretest_summary.md     ranked table + verdict
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

OUT = Path("out_faces_hc")

# ----------------------------------------------------------------- SHARD BUDGET
TOTAL_SHARDS = 300
N_WALLS = 2
SHARDS_PER_WALL = TOTAL_SHARDS // N_WALLS          # 150
GRID = int(round(math.sqrt(SHARDS_PER_WALL)))      # 12 cells across == 8.3% of head width
WORK = 256                                          # analysis resolution (square)
N_TONES = 5            # buildable `noir` palette: white + GRAY_L + GRAY_M + GRAY_D + K

# Detail = everything finer than the head mass itself.
HEAD_SCALE = 0.30      # sigma for the "head mass" lowpass, as a fraction of crop width

# Gates, set from the physics before looking at results.
DETAIL_KNEE = 0.30     # keep >=30% of face-box detail signal
BANDS_KNEE = 2         # brow/eye band AND mouth band must both still be detectable
DEPTH_KNEE = 0.15      # strongest band at least 15% of the face box's tonal range


_SIZE_CACHE: dict[str, tuple[int, int]] = {}


def image_size(path: str) -> tuple[int, int]:
    """(W, H) from the header only -- PIL does not decode pixels for `.size`."""
    if path not in _SIZE_CACHE:
        with Image.open(path) as im:
            _SIZE_CACHE[path] = im.size
    return _SIZE_CACHE[path]


@dataclass
class Cand:
    """BOTH boxes are fractions of the FULL source image, so they can be read straight off
    the labelled grid that `face_crop_check.py` draws. The old schema had `face` relative
    to the head crop, which meant every crop correction silently invalidated it."""
    key: str
    path: str
    group: str                 # "two_tone" | "oil_control"
    who: str
    head: tuple                # (l,t,r,b) of FULL image -> head-tight box, ANY aspect
    face_img: tuple            # (l,t,r,b) of FULL image -> brow..chin box
    note: str = ""

    @property
    def crop(self) -> tuple:
        """`head` grown about its centre to a SQUARE in PIXEL space. May extend OUTSIDE
        the image; `load_head` pads by edge-replication rather than clamping.

        Squaring by padding rather than by resizing is the point. `load_head` used to
        force the head box to a square with a non-uniform LANCZOS resize, which stretched
        Pearl 1.34x and Mona 1.19x horizontally -- the faces were literally the wrong
        shape before a single shard was cut. Clamping the square to the image bounds is
        no better: it re-crops the head (it silently sliced Mallarme's hair mass off top
        and bottom). Padding is the only option that cannot lose a pixel of the subject.
        Cost is background only, and `load_color_target` re-fits the subject to the wall
        downstream anyway."""
        W, H = image_size(self.path)
        l, t, r, b = self.head
        x0, y0, x1, y1 = l * W, t * H, r * W, b * H
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        return ((cx - side / 2) / W, (cy - side / 2) / H,
                (cx + side / 2) / W, (cy + side / 2) / H)

    @property
    def face(self) -> tuple:
        """`face_img` re-expressed as fractions of `crop` -- what every consumer wants."""
        cl, ct, cr, cb = self.crop
        fl, ft, fr, fb = self.face_img
        w, h = max(1e-6, cr - cl), max(1e-6, cb - ct)
        return (min(max((fl - cl) / w, 0.0), 1.0), min(max((ft - ct) / h, 0.0), 1.0),
                min(max((fr - cl) / w, 0.0), 1.0), min(max((fb - ct) / h, 0.0), 1.0))


# Boxes below were set by reading `out_faces_hc/crop/full_<key>.png` (labelled 0.05 grid),
# not by guessing. Re-run `python face_crop_check.py` after touching any of them.
CANDIDATES = [
    # --- TEST: stark two-tone woodcut faces (Vallotton, d.1925 -> public domain) -------
    Cand("poe", "examples/faces_hc/poe.jpg", "two_tone", "Edgar Allan Poe",
         (0.14, 0.13, 0.74, 0.66), (0.26, 0.36, 0.60, 0.62),
         "flat black/light masses, no hatching -- the best case for this theory"),
    Cand("dostoevsky", "examples/faces_hc/dostoevsky.jpg", "two_tone", "Dostoevsky",
         (0.17, 0.02, 0.82, 0.84), (0.28, 0.30, 0.72, 0.75),
         "literally 2-tone already: pure black + pure white"),
    Cand("mallarme", "examples/faces_hc/mallarme.jpg", "two_tone", "Mallarme",
         (0.10, 0.03, 0.92, 0.88), (0.36, 0.20, 0.74, 0.71),
         "huge black hair/beard mass, small light face island"),
    Cand("wagner", "examples/faces_hc/wagner.jpg", "two_tone", "Wagner",
         (0.16, 0.07, 0.90, 0.64), (0.22, 0.34, 0.62, 0.58),
         "high contrast but LINE-hatched, not flat mass"),
    Cand("ibsen", "examples/faces_hc/ibsen.jpg", "two_tone", "Ibsen",
         (0.18, 0.06, 0.75, 0.72), (0.31, 0.25, 0.65, 0.69),
         "high contrast but LINE-hatched, not flat mass"),
    # --- CONTROL: smooth oil-painting faces (the ~2750-shard failures) ----------------
    Cand("mona", "examples/mona_louvre_nobg_orig.png", "oil_control", "Mona Lisa",
         (0.29, 0.07, 0.65, 0.50), (0.37, 0.16, 0.55, 0.42), "smooth sfumato"),
    Cand("pearl", "examples/pearl_nobg_orig.png", "oil_control", "Girl w/ Pearl Earring",
         (0.32, 0.09, 0.70, 0.60), (0.36, 0.30, 0.57, 0.55), "smooth"),
    Cand("munch_self", "examples/munch_self_src.jpg", "oil_control", "Munch self",
         (0.36, 0.05, 0.58, 0.28), (0.42, 0.10, 0.53, 0.25), "smooth"),
    Cand("vangogh_self", "examples/vangogh_src.jpg", "oil_control", "Van Gogh self",
         (0.20, 0.05, 0.85, 0.75), (0.32, 0.18, 0.67, 0.67), "smooth, heavy texture"),
]


# ---------------------------------------------------------------------- HELPERS
def load_head(c: Cand, size: int = WORK, scale: float = 1.0) -> np.ndarray:
    """Square head crop -> `size` grayscale, normalised on the 2nd/98th percentiles.

    The square comes from `Cand.crop` (padded, aspect-preserving), so the resize below is
    uniform and the head is never stretched. `scale` grows/shrinks the box about its
    centre (nuisance transform for the grid-phase identity probe)."""
    img = Image.open(c.path).convert("RGB")
    W, H = img.size
    l, t, r, b = c.crop
    cx, cy = (l + r) / 2, (t + b) / 2
    hw, hh = (r - l) / 2 * scale, (b - t) / 2 * scale
    x0, y0 = int(round((cx - hw) * W)), int(round((cy - hh) * H))
    x1, y1 = int(round((cx + hw) * W)), int(round((cy + hh) * H))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    pad = ((max(0, -y0), max(0, y1 - H)), (max(0, -x0), max(0, x1 - W)), (0, 0))
    if any(p for axis in pad for p in axis):
        arr = np.pad(arr, pad, mode="edge")     # replicate the paper, do not invent black
        y0 += pad[0][0]; y1 += pad[0][0]
        x0 += pad[1][0]; x1 += pad[1][0]
    arr = arr[y0:y1, x0:x1]

    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)) \
               .resize((size, size), Image.LANCZOS)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    g = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    return np.clip((g - lo) / max(1e-6, hi - lo), 0.0, 1.0)


def posterize_gray(g: np.ndarray, k: int) -> np.ndarray:
    """Snap luminance to k buildable tones (1-D Lloyd-Max). k=2 -> the stark two-tone
    poster we would actually cut; k=N_TONES -> the `noir` perspex set."""
    lv = np.linspace(0.0, 1.0, k).astype(np.float32)
    for _ in range(30):
        idx = np.abs(g[..., None] - lv[None, None, :]).argmin(-1)
        new = np.array([g[idx == i].mean() if (idx == i).any() else lv[i]
                        for i in range(k)], dtype=np.float32)
        if np.allclose(new, lv, atol=1e-4):
            lv = new
            break
        lv = new
    idx = np.abs(g[..., None] - lv[None, None, :]).argmin(-1)
    return lv[idx].astype(np.float32)


def shard_simulate(g: np.ndarray, grid: int = GRID, k: int = N_TONES,
                   phase: float = 0.0) -> np.ndarray:
    """Simulate a `SHARDS_PER_WALL`-piece flat-shard reconstruction of this wall.

    Area-average onto a grid x grid lattice (one cell = one flat shard), snap each cell to
    the buildable palette, expand back. `phase` offsets the lattice by that fraction of a
    cell -- real shards do not align to anyone's eyes, so a result that only holds at
    phase 0 is an artifact of the simulation, not a property of the image.

    NOT an upper bound (the module banner records the disproof): a uniform lattice wastes
    cells on clear-white areas that cost the real solver no shards at all, and cannot put
    its cell edges on the ink boundary the way a subject-clipped Voronoi does."""
    H, W = g.shape
    sh = int(round(phase * H / grid))
    if sh:
        g = np.roll(g, (sh, sh), axis=(0, 1))
    small = np.asarray(Image.fromarray((g * 255).astype(np.uint8))
                       .resize((grid, grid), Image.BOX), dtype=np.float32) / 255.0
    small = posterize_gray(small, k)
    big = np.asarray(Image.fromarray((small * 255).astype(np.uint8))
                     .resize((W, H), Image.NEAREST), dtype=np.float32) / 255.0
    return np.roll(big, (-sh, -sh), axis=(0, 1)) if sh else big


def face_slice(g: np.ndarray, face: tuple) -> tuple[slice, slice]:
    H, W = g.shape
    l, t, r, b = face
    return slice(int(t * H), int(b * H)), slice(int(l * W), int(r * W))


def detail_of(g: np.ndarray) -> np.ndarray:
    """Everything finer than the head mass itself (high-pass at HEAD_SCALE)."""
    return g - ndimage.gaussian_filter(g, HEAD_SCALE * g.shape[1])


def detail_retention(orig: np.ndarray, sim: np.ndarray, face: tuple) -> float:
    """Fraction of FACE-BOX detail signal that survives, credited only where it stayed
    correlated with the source. Projection form, so cell-boundary noise scores ~0."""
    ys, xs = face_slice(orig, face)
    a = detail_of(orig)[ys, xs]
    b = detail_of(sim)[ys, xs]
    a = a - a.mean()
    b = b - b.mean()
    den = float((a * a).sum())
    return max(0.0, float((a * b).sum()) / den) if den > 1e-9 else 0.0


def feature_bands(g: np.ndarray, face: tuple) -> tuple[int, float, np.ndarray]:
    """Do the eyes/nose/mouth still land as distinguishable dark bands?

    Row-mean luminance profile down the face box. A face that still reads has a dark
    brow/eye band in the upper half and a dark mouth/moustache band in the lower half.
    Returns (n_bands, depth_of_strongest, profile). Depth is a minimum's prominence
    against its flanking maxima, as a fraction of the face box's tonal range."""
    ys, xs = face_slice(g, face)
    prof = g[ys, xs].mean(axis=1)
    prof = ndimage.uniform_filter1d(prof, max(3, len(prof) // 24))
    rng = float(prof.max() - prof.min())
    if rng < 1e-6:
        return 0, 0.0, prof
    n = len(prof)
    found = []
    for i in range(2, n - 2):
        if prof[i] <= prof[i - 1] and prof[i] <= prof[i + 1]:
            d = (min(prof[:i].max(), prof[i:].max()) - prof[i]) / rng
            if d > 0.05:
                found.append((float(d), i / n))
    found.sort(key=lambda t: -t[0])
    kept: list[tuple[float, float]] = []
    for d, pos in found:                       # merge minima <12% of box height apart
        if all(abs(pos - p) > 0.12 for _, p in kept):
            kept.append((d, pos))
    strong = [k for k in kept if k[0] >= DEPTH_KNEE]
    return len(strong), (kept[0][0] if kept else 0.0), prof


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


# ------------------------------------------------------------------------- MAIN
def main() -> None:
    OUT.mkdir(exist_ok=True)
    print("=" * 78)
    print("FACE PRE-TEST -- can a stark two-tone face survive 300 shards?")
    print("=" * 78)
    print(f"  budget      : {TOTAL_SHARDS} shards / {N_WALLS} walls = {SHARDS_PER_WALL} per wall")
    print(f"  ideal grid  : {GRID}x{GRID}  ->  1 shard = {100.0 / GRID:.1f}% of head width")
    print("  GRAYSCALE   : hue is discarded, so the colour-compatibility / colour-agreeing")
    print("                double-duty result does NOT transfer to any pair from this pool.\n")

    recs = []
    for c in CANDIDATES:
        if not Path(c.path).exists():
            print(f"  SKIP {c.key}: missing {c.path}")
            continue
        g = load_head(c)
        two = posterize_gray(g, 2)              # the stark poster we would cut from
        sim = shard_simulate(two)               # what 150 flat shards can show
        sim_smooth = shard_simulate(g)          # same budget straight off the smooth source

        n_b, depth, prof = feature_bands(sim, c.face)
        n_b0, depth0, prof0 = feature_bands(two, c.face)
        rec = dict(
            key=c.key, who=c.who, group=c.group, note=c.note, path=c.path,
            detail_retain=detail_retention(two, sim, c.face),
            detail_retain_smooth=detail_retention(g, sim_smooth, c.face),
            n_bands=n_b, band_depth=depth,
            n_bands_source=n_b0, band_depth_source=depth0,
        )
        rec["passing"] = bool(rec["detail_retain"] >= DETAIL_KNEE
                              and rec["n_bands"] >= BANDS_KNEE
                              and rec["band_depth"] >= DEPTH_KNEE)
        rec.update(_g=g, _two=two, _sim=sim, _prof=prof, _prof0=prof0, _cand=c)
        recs.append(rec)
        print(f"  {'PASS' if rec['passing'] else 'fail':4s}  "
              f"detail={rec['detail_retain']:.3f}  bands={n_b} (src {n_b0})  "
              f"depth={depth:.3f}   {c.who}  [{c.group}]")

    # ---- grid-phase rank-1 identity: probe = half-cell-shifted, 6%-rescaled sim -------
    gal = [r["_sim"] for r in recs]
    for i, r in enumerate(recs):
        probe = shard_simulate(posterize_gray(load_head(r["_cand"], scale=1.06), 2),
                               phase=0.5)
        sc = [ncc(probe, gg) for gg in gal]
        best = int(np.argmax(sc))
        r["id_rank1"] = bool(best == i)
        r["id_self"] = float(sc[i])
        r["id_best_other"] = float(max(s for j, s in enumerate(sc) if j != i))
        r["id_margin"] = r["id_self"] - r["id_best_other"]
        r["id_confused_with"] = recs[best]["who"] if best != i else None

    n_id = sum(r["id_rank1"] for r in recs)
    print(f"\n  grid-phase rank-1 identity: {n_id}/{len(recs)}  "
          f"(chance = {100.0 / len(recs):.0f}%)")

    contact_sheet(recs, OUT / "pretest_contact.png")
    write_summary(recs, OUT / "pretest_summary.md")
    (OUT / "pretest_scores.json").write_text(
        json.dumps([{k: v for k, v in r.items() if not k.startswith("_")} for r in recs],
                   indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT}/pretest_contact.png  <- the real evidence")
    print(f"  wrote {OUT}/pretest_summary.md, {OUT}/pretest_scores.json")


def contact_sheet(recs: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(recs)
    fig, axes = plt.subplots(4, n, figsize=(2.05 * n, 8.8),
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 0.62]))
    axes = np.atleast_2d(axes)
    for j, r in enumerate(recs):
        c = r["_cand"]
        for i, kk in enumerate(("_g", "_two", "_sim")):
            ax = axes[i, j]
            ax.imshow(r[kk], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            H, W = r[kk].shape
            fl, ft, fr, fb = c.face
            ax.add_patch(Rectangle((fl * W, ft * H), (fr - fl) * W, (fb - ft) * H,
                                   fill=False, ec="yellow", lw=0.8, ls=":"))
            if i == 0:
                col = "tab:blue" if r["group"] == "two_tone" else "tab:red"
                ax.set_title(f"{r['who']}\n[{r['group']}]", fontsize=7.5, color=col)
            if i == 2:
                ax.set_xlabel(f"{'PASS' if r['passing'] else 'FAIL'}  "
                              f"detail={r['detail_retain']:.2f}\n"
                              f"bands={r['n_bands']}/{r['n_bands_source']}  "
                              f"depth={r['band_depth']:.2f}  "
                              f"id={'Y' if r['id_rank1'] else 'N'}",
                              fontsize=7, color="green" if r["passing"] else "red")
        ax = axes[3, j]
        yy = np.linspace(1, 0, len(r["_prof0"]))
        ax.plot(r["_prof0"], yy, lw=1.0, color="0.6", label="2-tone")
        ax.plot(r["_prof"], np.linspace(1, 0, len(r["_prof"])), lw=1.4,
                color="crimson", label="300 shards")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel("face-box\nprofile", fontsize=8)
            ax.legend(fontsize=5.5, loc="lower left")
    for i, lbl in enumerate(("head crop (gray)", "stark 2-tone poster",
                             f"{SHARDS_PER_WALL} shards ({GRID}x{GRID})")):
        axes[i, 0].set_ylabel(lbl, fontsize=8)
    fig.suptitle(
        f"Face pre-test: does a stark two-tone face survive {TOTAL_SHARDS} shards?   "
        f"BLUE = high-contrast test group,  RED = smooth-oil control (the ~2750-shard failures).\n"
        f"Row 3 is an OPTIMISTIC upper bound: ideal uniform {GRID}x{GRID} shard grid, perfect "
        f"placement, {N_TONES}-tone noir palette. Real solver shards are irregular -> strictly worse.\n"
        f"Yellow box = face box (brow..chin); detail metrics are restricted to it so the "
        f"hair/silhouette edge cannot inflate them. Row 4 = luminance profile down that box.",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(recs: list[dict], out_path: Path) -> None:
    def grp(name):
        v = [r for r in recs if r["group"] == name]
        return dict(n=len(v),
                    detail=float(np.mean([r["detail_retain"] for r in v])),
                    bands=float(np.mean([r["n_bands"] for r in v])),
                    depth=float(np.mean([r["band_depth"] for r in v])))

    tt, oc = grp("two_tone"), grp("oil_control")
    lines = [
        "# Face pre-test -- do stark two-tone faces survive 300 shards?",
        "",
        "> **SUPERSEDED by `face_render300.py`.** This script's core claim -- that an ideal",
        "> uniform 12x12 shard grid is an *optimistic upper bound* -- is **false**. The real",
        "> solver render of Poe and Dostoevsky is cleanly recognisable (face-box detail 0.745,",
        "> at ~208 shards, under the 300 budget), while this script ranked the two-tone group",
        "> *below* the smooth-oil controls. A uniform grid wastes cells on clear-white areas",
        "> that cost the solver no shards, and cannot align its cell edges to the ink boundary.",
        "> Use the render, not this table, to screen images.",
        "",
        "## Caveat recorded up front",
        "",
        "This direction is **grayscale**. Hue is discarded, so the colour-compatibility /",
        "colour-agreeing double-duty result (`shadowart-noise.md`, `report_team.md`) does",
        "**not** transfer to any pair drawn from this pool. Choosing grayscale faces means",
        "giving up the double-duty argument on those walls. Flagged deliberately.",
        "",
        "## Method",
        "",
        f"- Budget {TOTAL_SHARDS} shards / {N_WALLS} walls = **{SHARDS_PER_WALL} per wall**; an ideal",
        f"  uniform tiling is a **{GRID}x{GRID}** grid, so one shard subtends **{100.0 / GRID:.1f}% of",
        "  head width**. Eyes, brows and mouth are all at or below that size -- which is why a",
        "  uniform-grid argument predicts failure. The real solver does not tile uniformly.",
        "- Row 3 of the contact sheet was *claimed* to be an optimistic upper bound. It is not;",
        "  see the banner above. It is closer to a lower bound for flat, hard-edged sources.",
        "- All detail metrics are restricted to a hand-set **face box** (brow-to-chin), because a",
        "  first pass over the whole head crop ranked the *Mona Lisa* top: at feature scale the",
        "  dominant energy is the long sharp edge of the hair/dress mass, which a coarse grid",
        "  reproduces well. That was measuring silhouette, not face.",
        f"- Gates set from the physics beforehand: `detail_retain >= {DETAIL_KNEE}`,",
        f"  `n_bands >= {BANDS_KNEE}`, `band_depth >= {DEPTH_KNEE}`.",
        "",
        "## Results",
        "",
        "| who | group | detail_retain | bands (sim/src) | band_depth | rank-1 id | margin | verdict |",
        "|-----|-------|--------------:|:---------------:|-----------:|:---------:|-------:|---------|",
    ]
    for r in sorted(recs, key=lambda r: -r["detail_retain"]):
        lines.append(
            f"| {r['who']} | {r['group']} | {r['detail_retain']:.3f} "
            f"| {r['n_bands']}/{r['n_bands_source']} | {r['band_depth']:.3f} "
            f"| {'Y' if r['id_rank1'] else 'N'} | {r['id_margin']:+.3f} "
            f"| {'PASS' if r['passing'] else 'FAIL'} |")
    lines += [
        "",
        "## Group means",
        "",
        "| group | n | detail_retain | n_bands | band_depth |",
        "|-------|--:|--------------:|--------:|-----------:|",
        f"| two_tone (test) | {tt['n']} | {tt['detail']:.3f} | {tt['bands']:.2f} | {tt['depth']:.3f} |",
        f"| oil_control | {oc['n']} | {oc['detail']:.3f} | {oc['bands']:.2f} | {oc['depth']:.3f} |",
        "",
        f"Two-tone advantage in face-box detail retention: **{tt['detail'] - oc['detail']:+.3f}**",
        "",
        "`pretest_contact.png` is the actual evidence; the numbers above are proxies.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
