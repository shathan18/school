"""THE DECISIVE NUMBER: how many shards does a face need when the piece REALLY intersects?

`face_paper_floor.py` established the trade in one table. With clear paper the two-tone face
is legible at ~200 shards (face_det 0.74) but the sculpture barely intersects (2/10 panels
serve both walls, cross-talk 1.3%, good 0.58%). Tint the paper GRAY_L and the sculpture
becomes real (9/10 panels, cross-talk 23.7%, good 17.5%, good/bad 1.86 -- better than any
COLOUR pair's ~0.9) but the face collapses to face_det 0.45.

So the original headline -- "high-contrast two-tone faces survive 300 shards where oils
needed ~2750" -- was measured on a configuration that is not an intersection sculpture. The
saving came from leaving 56% of the piece as bare wall, and bare wall cannot participate in
an intersection. It was two shadow puppets sharing a room.

This script asks the question properly: holding the intersection requirement FIXED (tinted
paper, every part of the piece is material), how does face legibility scale with shard
count, and where does it cross the bar the clear-paper run set (face_det ~0.74)?

DENSITY, NOT BUDGET. A first version of this scan swept `shard_budget` from 300 to 2750 and
got 343 shards at every single setting. `_autotune_spacing` documents why: the budget is a
fabrication CEILING that only ever COARSENS an overshooting layout, never refines one. Shard
count is set by `solve.fragment_size`, with `fragment_min_area` dropping anything smaller.
So the scan sweeps a linear `density` multiplier on both (areas as its square) and reports
the ACHIEVED count. Every x-axis value here is a measured shard count, not a requested one.

CONTROL: the same scan on the smooth-oil pair (Mona / Pearl), already ~85% ink and so barely
changed by tinting. It separates a genuine flat-mass advantage from an ink-area artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import face_render300 as FR
import face_paper_floor as PF

OUT = Path("out_faces_hc/duty")

DENSITIES = [1.0, 0.72, 0.52, 0.38, 0.28, 0.21]     # linear shard-size multipliers
LEGIBLE = 0.74                                       # clear-paper face_det, the bar to beat
BIG_BUDGET = 100000                                  # non-binding: let density set the count


def scan(pair, floor, arm, cands, seed=2):
    rows = []
    old = FR.SHARD_BUDGET_PER_WALL
    FR.SHARD_BUDGET_PER_WALL = BIG_BUDGET
    try:
        for d in DENSITIES:
            b = PF.build_floor(pair, seed, floor, arm, cands, density=d)
            m = PF.evaluate(b, pair, seed, floor, arm, density=d)
            rows.append(m)
            print(f"{pair[0][:24]:24s} {floor:>5.2f} {d:>5.2f} {m['shards']:>7d} | "
                  f"{m['good_mean']:>6.2f} {m['bad_mean']:>6.2f} {(m['good_bad'] or 0):>5.2f} "
                  f"{m['n_both']:>3d}/{m['n_panels']:<2d} | {m['ssim']:>5.3f} "
                  f"{m['face_det']:>5.3f}{'  <== LEGIBLE' if m['face_det'] >= LEGIBLE else ''}")
    finally:
        FR.SHARD_BUDGET_PER_WALL = old
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 104)
    print("DENSITY SCAN under a REAL intersection constraint (GRAY_L paper = all material)")
    print(f"  legibility bar: face_det >= {LEGIBLE} (what clear paper reached at ~200 shards)")
    print("=" * 104)
    print(f"{'pair':24s} {'floor':>5s} {'dens':>5s} {'shards':>7s} | {'good%':>6s} "
          f"{'bad%':>6s} {'g/b':>5s} {'both':>6s} | {'SSIM':>5s} {'face':>5s}")
    print("-" * 104)

    rows = []
    c75 = PF.write_floor_targets(0.75)
    rows += scan(FR.PAIRS[0], 0.75, "uniform", c75)          # two-tone, tinted -> real piece
    print()
    rows += scan(FR.PAIRS[2], 0.75, "uniform", c75)          # smooth-oil control, tinted
    print()
    c100 = PF.write_floor_targets(1.0)
    rows += scan(FR.PAIRS[0], 1.0, "uniform", c100)          # two-tone, clear paper (old)

    (OUT / "density_scan.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n--- crossing points (fewest ACHIEVED shards reaching the legibility bar) ---")
    for lab, fl, tag in [(FR.PAIRS[0][0], 0.75, "GRAY_L paper (REAL intersection)"),
                         (FR.PAIRS[2][0], 0.75, "GRAY_L paper (REAL intersection)"),
                         (FR.PAIRS[0][0], 1.0, "clear paper (trivial, 2/10 panels)")]:
        sel = [r for r in rows if r["label"] == lab and r["floor"] == fl
               and r["face_det"] >= LEGIBLE]
        if sel:
            best = f"{min(r['shards'] for r in sel)} shards"
        else:
            top = max((r["face_det"] for r in rows
                       if r["label"] == lab and r["floor"] == fl), default=0.0)
            best = f"NOT REACHED (best face_det {top:.3f})"
        print(f"  {lab[:26]:26s} {tag:34s} -> {best}")
    print(f"\nwrote {OUT}/density_scan.json")


if __name__ == "__main__":
    main()
