"""Does more material AND finer shards give both intersection and a readable mark?

`logo_render.py` found the tension: raising ink area raises planes-serving-both
(4 -> 6 -> 7) but wrecks the mark (IoU 0.93 -> 0.67 -> 0.57). The inverted_floor renders
are visibly BLOBBY -- the gear teeth are smaller than a shard. That is a resolution
failure, not an intersection failure, and resolution is the one thing we know how to buy.

So: sweep density on the two arms that carry material. If IoU recovers while
planes-serving-both holds, logos beat faces outright -- for faces the same sweep
SATURATED (face_det 0.35 -> 0.55 and no further, never reaching the clear-paper 0.74).

`density` scales solve.fragment_size linearly and the area bounds as its square.
shard_budget is a CEILING and cannot do this (measured: 343 shards at every budget).

Run:  python logo_density.py
Out:  out_logos/density/{scan.json, _scan.png}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shadowart import metrics as _metrics

import face_paper_floor as PF
import face_duty_sweep as DS
import logo_render as LR

OUT = Path("out_logos/density")
ARMS = ["inverted", "inverted_floor"]
DENSITIES = [1.0, 0.60, 0.40, 0.28]
SEED = 2


def run(arm: str, density: float) -> dict:
    ka, kb = f"{LR.PAIR[0]}_{arm}", f"{LR.PAIR[1]}_{arm}"
    cands = LR.write_arm_targets(arm)

    old = PF.FR.SHARD_BUDGET_PER_WALL
    PF.FR.SHARD_BUDGET_PER_WALL = 100000          # non-binding; density sets the count
    try:
        b = PF.build_floor((arm, ka, kb), SEED, 1.0, "uniform", cands,
                           panel_count=LR.PANEL_COUNT, angle_range=LR.ANGLE_RANGE,
                           density=density)
    finally:
        PF.FR.SHARD_BUDGET_PER_WALL = old

    base, rows = DS.ablate(b)
    n_both = sum(r["serves_A"] and r["serves_B"] for r in rows)
    n_dead = sum((not r["serves_A"]) and (not r["serves_B"]) for r in rows)
    m = PF.evaluate(b, (arm, ka, kb), SEED, 1.0, "uniform", density=density)
    acc = _metrics.evaluate_wall_accuracy(b["targets"], base)
    iou = {w: LR.mark_iou(b["targets"][w], base[w]) for w in ("A", "B")}

    rec = dict(arm=arm, density=density, shards=m["shards"], n_both=n_both, n_dead=n_dead,
               good=m["good_mean"], bad=m["bad_mean"], gb=m["good_bad"],
               ssim=float(np.mean([acc["A"]["ssim"], acc["B"]["ssim"]])),
               iou_A=iou["A"], iou_B=iou["B"],
               iou=float(np.mean(list(iou.values()))))
    print(f"  {arm:16s} d={density:<5.2f} shards={rec['shards']:>5d}  both={n_both}/10 "
          f" dead={n_dead}  good={rec['good']:>6.2f}%  g/b={rec['gb']:>5.2f} "
          f" SSIM={rec['ssim']:.3f}  IoU={rec['iou']:.3f}")
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("LOGO DENSITY SCAN -- can finer shards restore the mark without losing the "
          "intersection?")
    print("=" * 96)
    recs = []
    for arm in ARMS:
        print(f"\n--- {arm} ---")
        for d in DENSITIES:
            recs.append(run(arm, d))
    (OUT / "scan.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm, mk in zip(ARMS, ("o-", "s-")):
        r = [x for x in recs if x["arm"] == arm]
        ax[0].plot([x["shards"] for x in r], [x["iou"] for x in r], mk, label=arm)
        ax[1].plot([x["shards"] for x in r], [x["n_both"] for x in r], mk, label=arm)
    ax[0].axhline(0.926, ls="--", c="grey")
    ax[0].annotate("normal polarity at 155 shards (but only 4/10 planes)",
                   (0.02, 0.93), xycoords=("axes fraction", "data"), fontsize=7, va="bottom")
    ax[0].set_xscale("log"); ax[0].set_xlabel("shards"); ax[0].set_ylabel("mark IoU")
    ax[0].set_title("legibility vs shards"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].axhline(LR.MIN_BOTH, ls="--", c="red")
    ax[1].set_xscale("log"); ax[1].set_xlabel("shards")
    ax[1].set_ylabel("planes serving BOTH images")
    ax[1].set_title("intersection vs shards"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle("Faces saturated here and never recovered. Logos are the test of whether "
                 "that was the subject or the method.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "_scan.png", dpi=115, bbox_inches="tight")

    print("\n" + "=" * 96)
    print(f"{'arm':16s}{'dens':>6s}{'shards':>8s}{'both':>7s}{'good%':>8s}{'g/b':>7s}"
          f"{'SSIM':>7s}{'IoU':>7s}")
    print("-" * 96)
    for r in recs:
        print(f"{r['arm']:16s}{r['density']:>6.2f}{r['shards']:>8d}{r['n_both']:>4d}/10"
              f"{r['good']:>8.2f}{r['gb']:>7.2f}{r['ssim']:>7.3f}{r['iou']:>7.3f}")
    print(f"\nwrote {OUT}/_scan.png")


if __name__ == "__main__":
    main()
