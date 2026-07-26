"""Pair blur-survivors by palette compatibility.

For every ordered pair of survivors (from out_pair_selection/pretest_scores.jsonl),
compute a palette-compatibility score:

  compat = (1 - EMD_lab_norm) * shared_series_bonus * dominant_agreement

where
  EMD_lab_norm    = 1D optimal-transport cost between the two posterized palettes,
                    matched greedily in Lab, weighted by cluster fractions, normalised
                    by a reference dE (100). 0 = identical palettes, 1 = maximally
                    different.
  shared_series   = 1.15 if BOTH files come from examples/series/ (same source series
                    -> palette bias by construction), else 1.0. This mirrors the
                    "same-artist / same-series pair is ideal" heuristic from the
                    brief and from corrections_note.md sec 4.
  dominant_agree  = 1 + 0.15 if both palettes' TOP cluster is within dE 15 of each
                    other (matched grounds -> shared negative-space colour). 1.0 else.

Output: out_pair_selection/pair_scores.tsv   ranked pair table
        out_pair_selection/pair_palettes.png visual palette-side-by-side of top pairs
"""
from __future__ import annotations

import json
from pathlib import Path
from itertools import combinations

import numpy as np

OUT = Path("out_pair_selection")


def load_survivors() -> list[dict]:
    survivors = []
    with (OUT / "pretest_scores.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["passing"]:
                r["palette_lab"] = np.asarray(r["palette_lab"], dtype=np.float32)
                r["palette_rgb"] = np.asarray(r["palette_rgb"], dtype=np.float32)
                r["palette_frac"] = np.asarray(r["palette_frac"], dtype=np.float32)
                survivors.append(r)
    return survivors


def palette_emd(a: dict, b: dict) -> float:
    """Approximate 1-D EMD in Lab between two weighted palettes: greedy match on
    smallest pairwise dE, transporting min(residual_A, residual_B) each step."""
    A_lab, A_w = a["palette_lab"].copy(), a["palette_frac"].astype(np.float32).copy()
    B_lab, B_w = b["palette_lab"].copy(), b["palette_frac"].astype(np.float32).copy()
    # normalise weights (should already sum to ~1)
    A_w = A_w / max(1e-9, A_w.sum())
    B_w = B_w / max(1e-9, B_w.sum())
    # pairwise distance matrix
    D = np.linalg.norm(A_lab[:, None, :] - B_lab[None, :, :], axis=-1)   # (nA, nB)
    total = 0.0
    while A_w.sum() > 1e-6 and B_w.sum() > 1e-6:
        i, j = np.unravel_index(np.argmin(D + (A_w[:, None] <= 0) * 1e9
                                            + (B_w[None, :] <= 0) * 1e9), D.shape)
        flow = float(min(A_w[i], B_w[j]))
        if flow <= 0:
            break
        total += flow * float(D[i, j])
        A_w[i] -= flow
        B_w[j] -= flow
    return total   # in dE units, weighted


def dominant_dE(a: dict, b: dict) -> float:
    return float(np.linalg.norm(a["palette_lab"][0] - b["palette_lab"][0]))


def is_series(row: dict) -> bool:
    return "/series/" in row["path"]


def score_pair(a: dict, b: dict) -> dict:
    emd = palette_emd(a, b)               # weighted dE, typically 5..80
    emd_norm = min(1.0, emd / 100.0)      # 100 dE = maximally different -> 1.0
    palette_match = 1.0 - emd_norm         # 0..1
    shared = 1.15 if (is_series(a) and is_series(b)) else 1.0
    dom_bonus = 1.15 if dominant_dE(a, b) < 15.0 else 1.0
    compat = palette_match * shared * dom_bonus
    # survivor strength: geometric mean of the two blur scores (rewards two-way survivability)
    surv = (a["score"] * b["score"]) ** 0.5
    combined = compat * surv
    return dict(
        a=a["name"], b=b["name"],
        palette_emd=emd,
        palette_match=palette_match,
        shared_series=shared,
        dominant_dE=dominant_dE(a, b),
        dom_bonus=dom_bonus,
        surv_geomean=float(surv),
        compat=float(compat),
        combined=float(combined),
        pa=a["path"], pb=b["path"],
    )


def dedupe(survivors: list[dict]) -> list[dict]:
    """Fold obvious near-duplicates (same subject different filename) into one row.
    Keys chosen by inspection of the survivor set."""
    dup_map = {
        # red_fuji.jpg is the standalone crop of the series file
        "red_fuji": "red_fuji_southern_wind_clear_morning",
    }
    kept: dict[str, dict] = {}
    for r in survivors:
        canonical = dup_map.get(r["name"], r["name"])
        if canonical not in kept or r["score"] > kept[canonical]["score"]:
            kept[canonical] = r
    return list(kept.values())


def palette_swatch_png(survivors: list[dict], pairs: list[dict], out_path: Path, top_n: int = 8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    survs_by_name = {r["name"]: r for r in survivors}
    pairs = sorted(pairs, key=lambda p: -p["combined"])[:top_n]
    fig, axes = plt.subplots(top_n, 2, figsize=(10, top_n * 1.2))
    if top_n == 1:
        axes = np.atleast_2d(axes)
    for row, pr in enumerate(pairs):
        for col, key in enumerate(("a", "b")):
            r = survs_by_name[pr[key]]
            palette = r["palette_rgb"]        # (K, 3)
            fracs = r["palette_frac"]
            # draw a horizontal bar of coloured segments proportional to fracs
            ax = axes[row, col]
            x = 0.0
            for c in range(len(palette)):
                w = float(fracs[c])
                ax.add_patch(plt.Rectangle((x, 0), w, 1, color=np.clip(palette[c], 0, 1)))
                x += w
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
            title = pr["a"] if col == 0 else pr["b"]
            ax.set_title(title[:40], fontsize=7)
        axes[row, 0].text(-0.25, 0.5,
                          f"#{row+1}\ncompat={pr['compat']:.2f}\nEMD={pr['palette_emd']:.1f}\ncomb={pr['combined']:.2f}",
                          transform=axes[row, 0].transAxes, ha="right", va="center", fontsize=7)
    fig.suptitle(f"Top {top_n} palette-compatible pairs (survivors only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    survivors = load_survivors()
    print(f"survivors: {len(survivors)}")
    survivors = dedupe(survivors)
    print(f"after de-dup: {len(survivors)}")
    pairs = [score_pair(a, b) for a, b in combinations(survivors, 2)]
    pairs.sort(key=lambda p: -p["combined"])

    # TSV
    tsv = OUT / "pair_scores.tsv"
    with tsv.open("w", encoding="utf-8") as fh:
        fh.write("rank\ta\tb\tcombined\tcompat\tpalette_match\tpalette_emd\tdominant_dE\t"
                 "shared_series\tsurv_geomean\tpath_a\tpath_b\n")
        for i, p in enumerate(pairs, 1):
            fh.write(f"{i}\t{p['a']}\t{p['b']}\t{p['combined']:.3f}\t{p['compat']:.3f}\t"
                     f"{p['palette_match']:.3f}\t{p['palette_emd']:.2f}\t{p['dominant_dE']:.1f}\t"
                     f"{p['shared_series']:.2f}\t{p['surv_geomean']:.3f}\t{p['pa']}\t{p['pb']}\n")
    print(f"wrote {tsv} with {len(pairs)} pairs")

    palette_swatch_png(survivors, pairs, OUT / "pair_palettes.png", top_n=8)
    print(f"wrote {OUT / 'pair_palettes.png'}")

    print("\nTop 8 pairs:")
    for i, p in enumerate(pairs[:8], 1):
        print(f"  #{i}  combined={p['combined']:.3f}  compat={p['compat']:.3f}  "
              f"EMD={p['palette_emd']:5.1f}  domdE={p['dominant_dE']:5.1f}  "
              f"surv={p['surv_geomean']:.3f}  |  {p['a']}  x  {p['b']}")


if __name__ == "__main__":
    main()
