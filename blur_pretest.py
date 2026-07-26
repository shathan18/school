"""Blur pre-test: for each candidate artwork, apply a strong Gaussian blur + K-means
posterize to ~6 colours and score how much of a bold-single-shape signal survives.

CRITERION (per lecturer): at ~300 flat shards, an image survives only if it reduces to
one dominant bold shape on a contrasting ground with few colours. Red Fuji passed (red
triangle / blue sky / white base). Great Wave failed (mid-scale detail, no dominant
silhouette). We now select images by this MEASURED property, not by fame.

WHAT WE MEASURE (after downscale to ~256px, Gaussian sigma=12 on that, K-means K=6 in
CIELAB, then evaluate on the posterized image):

  1. n_big_regions   count of connected components (across ALL palette clusters) with
                     area >= 2% of frame. Bold posters have a small handful (3..7);
                     busy prints have 20+. LOW is better.
  2. top2_frac       area of the two biggest regions summed, as fraction of frame. A
                     Red-Fuji-style poster has ~80-95% (mountain + sky = the whole
                     picture). Busy prints spread mass across many small regions.
  3. subj_compact    isoperimetric ratio 4*pi*A / P^2 of the biggest NON-BRIGHT region
                     (the silhouette). 1.0 = disc, ~0.5 = triangle/wedge, ~0.05 =
                     spidery. HIGH is better.
  4. subj_area       area of that non-bright silhouette as fraction of frame. Should
                     be in [0.10, 0.75] - too small = blurred to nothing, too big =
                     no negative space.
  5. contrast_dE     CIELAB dE between the palette's brightest cluster and the darkest
                     large cluster. HIGH = the silhouette stands off the ground.
  6. palette_used    clusters with >=2% coverage. Fewer = poster-like.

Composite score (higher = better survivor):
   S = subj_compact
       * min(1.5, contrast_dE / 30.0)          # saturating at dE 45
       * top2_frac                              # posters dominate the frame
       * (5.0 / max(3, n_big_regions))         # penalise fragmentation
       * area_bonus(subj_area)                  # zero outside [0.05, 0.80]

Passing threshold (all four must hold): S >= 0.45, subj_compact >= 0.15,
n_big_regions <= 8, 0.08 <= subj_area <= 0.75. Knees set by inspecting Red Fuji /
fuji_a / h_amida_falls / kajikazawa (target survivors) vs Great Wave / Sumida /
Yoshida (target rejections).

CANDIDATE POOL (all secular landscapes, no women / religion / politics / mythology):
- Every Hokusai "36 Views of Mount Fuji" print already in examples/series/
- The three standalone bold Fuji cuts (fuji_a/b/c_m.png, red_fuji.jpg)
- Hokusai bold single-subject waterfalls (h_amida_falls, h_kirifuri_falls)
- Standalone series prints (enoshima, kajikazawa) that are also single-form
Excluded from series/: asakusa_honganji_temple, sazai_hall - Buddhist temples.

Outputs:
  out_pair_selection/pretest_contact.png     contact sheet (orig | blurred posterize)
  out_pair_selection/pretest_scores.jsonl    per-candidate metrics + PASS/FAIL
  out_pair_selection/pretest_summary.md      ranked table
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans

# -------------------------------------------------------------------- CONFIG
SIZE = 256                 # long-edge downscale target
BLUR_SIGMA = 12.0          # heavy Gaussian on the 256-px image (~19% of long edge)
K_COLOURS = 4              # posterize palette size (matches "reads with ~4 flat inks")
MIN_REGION_FRAC = 0.02     # connected components below this are ignored
COMPACT_KNEE = 0.20        # subj_compact >= to pass
N_REGIONS_MAX = 7          # n_big_regions <= to pass
SUBJ_AREA_LO = 0.08        # subj_area >= to pass
SUBJ_AREA_HI = 0.75        # subj_area <= to pass
SCORE_KNEE = 0.35          # composite S >= to pass
OUT = Path("out_pair_selection")

# Explicit exclusions - respect the "no women / religion / politics / mythology" rule.
# Even if the file is in the pool, drop it here.
EXCLUDE = {
    "asakusa_honganji_temple_in_th_eastern_capital",   # Buddhist temple (religion)
    "sazai_hall_at_the_temple_of_the_five_hundred_arh",  # Buddhist temple (religion)
    "besneeuwde_ochtend_in_koishikawa_rijksmuseum_ak_",  # tea-house tourist scene, snow field w/ people
}

# --------------------------------------------------------------------- POOL
def _series_files() -> list[Path]:
    root = Path("examples/series")
    return sorted(p for p in root.glob("*.jpg") if p.stem not in EXCLUDE)


def _extra_bold_fuji() -> list[Path]:
    """Standalone Hokusai bold cuts already in examples/."""
    names = [
        "red_fuji.jpg",
        "fuji_a.jpg", "fuji_b.jpg", "fuji_c.jpg",
        "h_amida_falls.jpg",       # bold waterfall column
        "h_kirifuri_falls.jpg",    # bold waterfall column
        "enoshima.jpg",             # bold headland silhouette
        "kajikazawa.jpg",           # bold rock+fisherman silhouette
    ]
    root = Path("examples")
    return [root / n for n in names if (root / n).exists()]


def candidate_pool() -> list[Path]:
    seen: dict[str, Path] = {}
    for p in _series_files() + _extra_bold_fuji():
        # Prefer the specific standalone files over series duplicates by stem prefix,
        # but keep both if they're clearly distinct.
        seen.setdefault(p.stem, p)
    return list(seen.values())


# ------------------------------------------------------------------- HELPERS
def load_rgb(path: Path, size: int = SIZE) -> np.ndarray:
    """Load image, correct EXIF, resize so long edge = size, return float RGB in [0,1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = size / max(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Approximate sRGB -> CIELAB (D65), no external dep. Good enough for clustering
    and dE deltas. rgb in [0,1]. Returns Lab with L in [0,100]."""
    # sRGB companding
    a = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    # Linear sRGB -> XYZ (D65)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = a @ M.T
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    xyz = xyz / np.array([Xn, Yn, Zn], dtype=np.float32)
    delta = 6/29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4/29)
    L = 116 * f[..., 1] - 16
    a_ = 500 * (f[..., 0] - f[..., 1])
    b_ = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a_, b_], axis=-1)


def gaussian_blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    """Per-channel Gaussian blur (edge-reflect)."""
    return np.stack([ndimage.gaussian_filter(rgb[..., c], sigma, mode="reflect")
                     for c in range(3)], axis=-1)


def posterize_kmeans(rgb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-means in CIELAB. Returns (labels HxW, palette_rgb Kx3, palette_lab Kx3)."""
    lab = rgb_to_lab(rgb).reshape(-1, 3)
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(lab)
    labels = km.labels_.reshape(rgb.shape[:2])
    # Recover per-cluster mean RGB (average of source pixels in each cluster)
    palette_rgb = np.zeros((k, 3), dtype=np.float32)
    for c in range(k):
        m = labels == c
        if m.any():
            palette_rgb[c] = rgb[m].mean(0)
    return labels, palette_rgb, km.cluster_centers_.astype(np.float32)


def score_candidate(rgb: np.ndarray) -> dict:
    blurred = gaussian_blur(rgb, BLUR_SIGMA)
    labels, pal_rgb, pal_lab = posterize_kmeans(blurred, K_COLOURS)
    H, W = labels.shape
    total = H * W

    # cluster sizes and palette usage
    sizes = np.bincount(labels.ravel(), minlength=K_COLOURS)
    order = np.argsort(-sizes)
    palette_used = int((sizes / total >= 0.02).sum())

    # enumerate ALL significant connected components across every cluster
    regions_all = []   # (cluster_id, area_frac, mask)
    for c in range(K_COLOURS):
        lbl, n = ndimage.label(labels == c)
        if n == 0:
            continue
        comp_sizes = np.bincount(lbl.ravel())[1:]
        for i, sz in enumerate(comp_sizes, 1):
            af = sz / total
            if af >= MIN_REGION_FRAC:
                regions_all.append((c, float(af), lbl == i))

    n_big_regions = len(regions_all)
    regions_all.sort(key=lambda r: -r[1])
    largest = regions_all[0][1] if regions_all else 0.0
    second = regions_all[1][1] if len(regions_all) > 1 else 0.0
    top2_frac = largest + second

    # brightest cluster = "ground" (usually sky/white paper); silhouette = biggest
    # region belonging to any OTHER cluster.
    brightest_c = int(np.argmax(pal_lab[:, 0]))
    subj_regs = [r for r in regions_all if r[0] != brightest_c]
    if subj_regs:
        subj_c, subj_area, subj_mask = subj_regs[0]
        m = subj_mask.astype(np.uint8)
        perim = int(((m[:, 1:] != m[:, :-1]).sum()
                     + (m[1:, :] != m[:-1, :]).sum()))
        area = int(m.sum())
        subj_compact = min(1.0, (4 * math.pi * area) / (perim ** 2)) if perim > 0 else 0.0
        # dE from bright ground to the subject cluster
        dE_bg = float(np.linalg.norm(pal_lab[brightest_c] - pal_lab[subj_c]))
    else:
        subj_c = -1
        subj_area = 0.0
        subj_compact = 0.0
        dE_bg = 0.0

    def _area_bonus(x: float) -> float:
        # peaks at 0.30, fades to 0 at 0.05 and 0.85
        if x <= 0.05 or x >= 0.85:
            return 0.0
        return max(0.0, 1.0 - ((x - 0.30) / 0.40) ** 2)

    score = (
        subj_compact
        * min(1.5, dE_bg / 30.0)
        * top2_frac
        * (5.0 / max(3, n_big_regions))
        * _area_bonus(subj_area)
    )

    passing = bool(
        score >= SCORE_KNEE
        and subj_compact >= COMPACT_KNEE
        and n_big_regions <= N_REGIONS_MAX
        and SUBJ_AREA_LO <= subj_area <= SUBJ_AREA_HI
    )

    pal_rgb_ordered = pal_rgb[order]
    pal_lab_ordered = pal_lab[order]
    sizes_ordered = sizes[order] / total

    return dict(
        n_big_regions=int(n_big_regions),
        top2_frac=float(top2_frac),
        subj_area=float(subj_area),
        subj_compact=float(subj_compact),
        contrast_dE=float(dE_bg),
        palette_used=int(palette_used),
        score=float(score),
        passing=passing,
        _labels=labels,
        _palette_rgb=pal_rgb,
        _palette_ordered_rgb=pal_rgb_ordered,
        _palette_ordered_lab=pal_lab_ordered,
        _palette_ordered_frac=sizes_ordered,
        _brightest_c=brightest_c,
    )


def render_posterized(labels: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    return palette_rgb[labels]


# ------------------------------------------------------------------- OUTPUT
def contact_sheet(records: list[dict], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    cols = 4         # 4 candidates per row -> orig + poster + score panel = 3 cols each
    # Actually simpler: 2 panels per candidate (orig, poster). 4 candidates per row.
    per_row = 4
    rows = math.ceil(n / per_row)
    fig, axes = plt.subplots(rows * 2, per_row, figsize=(per_row * 3, rows * 6))
    if rows == 1:
        axes = np.atleast_2d(axes)
    for i, r in enumerate(records):
        rr = (i // per_row) * 2
        cc = i % per_row
        ax_o = axes[rr, cc]
        ax_p = axes[rr + 1, cc]
        ax_o.imshow(r["_orig"])
        ax_o.set_title(r["name"], fontsize=7)
        ax_o.axis("off")
        poster = render_posterized(r["_labels"], r["_palette_rgb"])
        ax_p.imshow(np.clip(poster, 0, 1))
        tag = "PASS" if r["passing"] else "fail"
        colour = "green" if r["passing"] else "red"
        ax_p.set_title(
            f"{tag}  S={r['score']:.2f}\n"
            f"nreg={r['n_big_regions']} top2={r['top2_frac']:.2f}\n"
            f"cmp={r['subj_compact']:.2f} dE={r['contrast_dE']:.0f} "
            f"area={r['subj_area']:.2f}",
            fontsize=7, color=colour,
        )
        ax_p.axis("off")
    # blank any leftover axes
    for j in range(n, rows * per_row):
        rr = (j // per_row) * 2
        cc = j % per_row
        axes[rr, cc].axis("off")
        axes[rr + 1, cc].axis("off")
    fig.suptitle(f"Blur pre-test ({SIZE}px, sigma={BLUR_SIGMA}, K={K_COLOURS}).  "
                 f"PASS = one bold shape survives.  n={n}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def summary_md(records: list[dict], out_path: Path):
    passing = [r for r in records if r["passing"]]
    failing = [r for r in records if not r["passing"]]
    passing.sort(key=lambda r: -r["score"])
    failing.sort(key=lambda r: -r["score"])
    lines = [
        "# Blur pre-test results",
        "",
        f"Pool: {len(records)} candidates.  Passed: **{len(passing)}**.  Failed: {len(failing)}.",
        "",
        f"Config: downsample to {SIZE}px long edge, Gaussian sigma={BLUR_SIGMA}, "
        f"K-means K={K_COLOURS} in CIELAB.",
        f"Pass gates: score>={SCORE_KNEE}, subj_compact>={COMPACT_KNEE}, "
        f"n_big_regions<={N_REGIONS_MAX}, subj_area in [{SUBJ_AREA_LO}, {SUBJ_AREA_HI}].",
        "",
        "## Survivors (ranked)",
        "",
        "| # | name | score | subj_compact | n_big_regions | top2_frac | contrast dE | subj_area | palette used |",
        "|---|------|------:|-------------:|--------------:|----------:|------------:|----------:|-------------:|",
    ]
    for i, r in enumerate(passing, 1):
        lines.append(f"| {i} | {r['name']} | {r['score']:.3f} | {r['subj_compact']:.2f} | "
                     f"{r['n_big_regions']} | {r['top2_frac']:.2f} | "
                     f"{r['contrast_dE']:.1f} | {r['subj_area']:.2f} | {r['palette_used']} |")
    lines += ["", "## Rejected (ranked by score, best-first)", "",
              "| name | score | subj_compact | n_big_regions | top2_frac | contrast dE | subj_area | reason |",
              "|------|------:|-------------:|--------------:|----------:|------------:|----------:|--------|"]
    for r in failing:
        why = []
        if r["score"] < SCORE_KNEE: why.append(f"S<{SCORE_KNEE}")
        if r["subj_compact"] < COMPACT_KNEE: why.append(f"cmp<{COMPACT_KNEE}")
        if r["n_big_regions"] > N_REGIONS_MAX: why.append(f"nreg>{N_REGIONS_MAX}")
        if not (SUBJ_AREA_LO <= r["subj_area"] <= SUBJ_AREA_HI): why.append("area oob")
        lines.append(f"| {r['name']} | {r['score']:.3f} | {r['subj_compact']:.2f} | "
                     f"{r['n_big_regions']} | {r['top2_frac']:.2f} | "
                     f"{r['contrast_dE']:.1f} | {r['subj_area']:.2f} | {' '.join(why)} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------- MAIN
def main():
    OUT.mkdir(exist_ok=True)
    pool = candidate_pool()
    print(f"pool: {len(pool)} candidates")
    records = []
    for i, p in enumerate(pool, 1):
        try:
            rgb = load_rgb(p)
        except Exception as e:
            print(f"  [{i:>2}/{len(pool)}] SKIP {p.name}: {e}")
            continue
        s = score_candidate(rgb)
        s["name"] = p.stem
        s["path"] = str(p).replace("\\", "/")
        s["_orig"] = rgb
        records.append(s)
        tag = "PASS" if s["passing"] else "fail"
        print(f"  [{i:>2}/{len(pool)}] {tag:4s}  S={s['score']:.3f}  "
              f"cmp={s['subj_compact']:.2f} nreg={s['n_big_regions']} "
              f"top2={s['top2_frac']:.2f} dE={s['contrast_dE']:.0f} "
              f"area={s['subj_area']:.2f}  {p.stem}")

    # JSONL: keep only serialisable fields (drop the numpy previews)
    with (OUT / "pretest_scores.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            # keep the ordered palette so the pairing script can reuse it
            row["palette_rgb"] = r["_palette_ordered_rgb"].tolist()
            row["palette_lab"] = r["_palette_ordered_lab"].tolist()
            row["palette_frac"] = r["_palette_ordered_frac"].tolist()
            fh.write(json.dumps(row) + "\n")

    contact_sheet(records, OUT / "pretest_contact.png")
    summary_md(records, OUT / "pretest_summary.md")

    n_pass = sum(1 for r in records if r["passing"])
    print(f"\nWrote {OUT}/pretest_contact.png,  {OUT}/pretest_scores.jsonl,  "
          f"{OUT}/pretest_summary.md")
    print(f"Survivors: {n_pass}/{len(records)}")


if __name__ == "__main__":
    main()
