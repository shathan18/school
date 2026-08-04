"""Parameter sweeps over the Pearl-Girl turntable build, logged so they can be re-read.

Every evaluation is appended to `runs.jsonl` and memoised on the parameter key EXCLUDING
the seed, so a sweep can be interrupted and resumed, and so repeated points cost nothing.
Scores are averaged over `--seeds` layout seeds: the shard layout alone moves the objective
by ~0.03 IoU, which is the same size as most of the effects under test, so a single-seed
ranking is not a ranking at all.

Ranking is on ALL-LIGHT mean IoU, never on the shards-only number -- shards-only deletes
the cross-talk, which is the very thing these sweeps exist to reduce.

    python pearl3_sweep.py crosstalk --arm 30fit
    python pearl3_sweep.py resolution --arm 30fit
    python pearl3_sweep.py magnification
    python pearl3_sweep.py density --arm 30fit
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from pearl3_baseline import ARMS, BuildConfig, run_build
from shadowart.targets import engrave


def _key(overrides: Dict) -> str:
    return json.dumps(overrides, sort_keys=True, default=str)


class Sweep:
    """Runs a grid of config overrides, memoised and journalled."""

    def __init__(self, base: BuildConfig, out: Path, seeds: Sequence[int]):
        self.base, self.out, self.seeds = base, out, list(seeds)
        self.out.mkdir(parents=True, exist_ok=True)
        self.journal = self.out / "runs.jsonl"
        self.cache: Dict[str, dict] = {}
        if self.journal.exists():
            for line in self.journal.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if "score" not in r:            # journal predates the composite score
                        r["score"] = 0.25 * r["mean_iou"] + 0.25 * r["min_iou"] + 0.50 * r["ssim"]
                    self.cache[r["key"]] = r

    def evaluate(self, overrides: Dict) -> dict:
        key = _key(overrides)
        if key in self.cache:
            return self.cache[key]
        runs = []
        for s in self.seeds:
            cfg = dataclasses.replace(self.base, seed=s, **overrides)
            r = run_build(cfg, self.out / "_scratch", verbose=False)
            runs.append(r)
        rec = {
            "key": key, "overrides": overrides, "seeds": self.seeds,
            "mean_iou": float(np.mean([r["summary"]["mean_iou"] for r in runs])),
            "min_iou": float(np.mean([r["summary"]["min_iou"] for r in runs])),
            "std_mean_iou": float(np.std([r["summary"]["mean_iou"] for r in runs])),
            "shards_only_iou": float(np.mean([r["summary_shards_only"]["mean_iou"] for r in runs])),
            "crosstalk_cost": float(np.mean([r["crosstalk_cost_iou"] for r in runs])),
            "distinctness": float(np.mean([r["summary"]["distinctness"] for r in runs])),
            "ssim": float(np.mean([np.mean([r["views"][v]["ssim"] for v in r["views"]])
                                   for r in runs])),
            "rmse": float(np.mean([np.mean([r["views"][v]["rmse"] for v in r["views"]])
                                   for r in runs])),
            "edge": float(np.mean([np.mean([r["views"][v]["edge_fidelity"] for v in r["views"]])
                                   for r in runs])),
            # Foreground area is the polarity/over-darkening tell: cross-talk inflates it
            # above the target's, and that inflation is what the noise work has to remove.
            "fg_pred": float(np.mean([np.mean([r["views"][v]["pred_fg_frac"] for v in r["views"]])
                                      for r in runs])),
            "fg_target": float(np.mean([np.mean([r["views"][v]["target_fg_frac"] for v in r["views"]])
                                        for r in runs])),
            "n_shards": float(np.mean([r["shards"]["n_fragments"] for r in runs])),
            "duty_all": float(np.mean([r["duty"]["frac_all"] for r in runs])),
            "runtime_s": float(np.sum([r["runtime_s"] for r in runs])),
        }
        # IoU alone is a silhouette score: it is blind to a reconstruction that has the right
        # outline but is far too dark inside it, which is exactly what these builds do. The
        # composite keeps the worst view honest (min IoU), so a build cannot buy a good mean
        # by sacrificing one stop, and gives tone half the say via SSIM.
        rec["score"] = float(0.25 * rec["mean_iou"] + 0.25 * rec["min_iou"]
                             + 0.50 * rec["ssim"])
        with self.journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.cache[key] = rec
        return rec

    def grid(self, axes: Dict[str, Sequence], label: str) -> List[dict]:
        keys = list(axes)
        combos = list(itertools.product(*(axes[k] for k in keys)))
        print(f"\n=== {label}: {len(combos)} configs x {len(self.seeds)} seeds ===")
        rows = []
        t0 = time.perf_counter()
        for i, vals in enumerate(combos, 1):
            ov = dict(zip(keys, vals))
            rec = self.evaluate(ov)
            rows.append(rec)
            desc = "  ".join(f"{k}={v}" for k, v in ov.items())
            print(f"[{i:3d}/{len(combos)}] {desc:52s} "
                  f"mean {rec['mean_iou']:.3f}+-{rec['std_mean_iou']:.3f}  "
                  f"min {rec['min_iou']:.3f}  xtalk {rec['crosstalk_cost']:.3f}  "
                  f"fg {rec['fg_pred']:.2f}/{rec['fg_target']:.2f}  "
                  f"n {rec['n_shards']:.0f}")
        print(f"  ({time.perf_counter() - t0:.0f}s)")
        return rows


def report(rows: List[dict], baseline: dict | None, top: int = 8, rank: str = "score") -> None:
    rows = sorted(rows, key=lambda r: -r[rank])
    print(f"\n  {'rank':<5}{'score':>7}{'mean':>7}{'min':>7}{'xtalk':>7}{'dist':>7}{'ssim':>7}"
          f"{'edge':>7}{'rmse':>7}{'fg':>6}{'n':>6}   overrides")
    for i, r in enumerate(rows[:top], 1):
        print(f"  {i:<5}{r['score']:>7.3f}{r['mean_iou']:>7.3f}{r['min_iou']:>7.3f}"
              f"{r['crosstalk_cost']:>7.3f}{r['distinctness']:>7.3f}{r['ssim']:>7.3f}"
              f"{r['edge']:>7.3f}{r['rmse']:>7.4f}{r['fg_pred']:>6.2f}{r['n_shards']:>6.0f}   "
              + "  ".join(f"{k}={v}" for k, v in r["overrides"].items()))
    if baseline is not None:
        d = rows[0][rank] - baseline[rank]
        verdict = "IMPROVES" if d > 2 * baseline["std_mean_iou"] else "within seed noise"
        print(f"\n  best vs baseline {rank} {baseline[rank]:.3f}: {d:+.3f} "
              f"({verdict}; seed sigma {baseline['std_mean_iou']:.3f})")


# Each sweep is one hypothesis. `None` in an axis means "leave the shipped default".
SWEEPS = {
    # Does pricing a shard's stray shadow at host-selection time reduce the cross-talk?
    # Memory says damage_weight without a credit term kills shard duty, so the credit is
    # swept jointly rather than after.
    "crosstalk": dict(damage_weight=[0.0, 0.25, 0.5, 1.0, 2.0],
                      credit_weight=[None, 0.25, 0.5, 1.0]),
    # `damage_weight<=0` is not "no penalty" -- it is a DIFFERENT branch that assigns hosts
    # at random, weighted by coverage. Any positive value switches to a deterministic argmax
    # over `cover - w*damage`, and because coverage barely varies between near-parallel
    # sheets while damage varies a lot, that argmax saturates to argmin(damage) almost
    # immediately. This sweep looks below the plateau for the weight where coverage still
    # has a say.
    "damage_fine": dict(damage_weight=[0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]),
    # Choosing each shard's TONE jointly with its host, instead of fixing it from its own
    # view first. The 4-level grey palette makes this cheaper than it was for colour.
    "blend": dict(colour_blend=[0.0, 0.25, 0.5, 0.75],
                  colour_primary_tol=[0.10, 0.25]),
    # Is the 30x30 detail gap actually resolution-limited? On this GPU 600px costs the same
    # as 300px, so if this curve is flat the gap is not resolution and the answer is noise.
    "resolution": dict(wall_res=[300, 450, 600, 800], panel_res=[300, 600]),
    # The core 30x30 trade: a smaller projected image means lower magnification, which
    # shrinks every fabrication limit and every penumbra on the wall. This measures the
    # fidelity-vs-image-size frontier rather than assuming a point on it.
    "magnification": dict(image=[0.90, 1.20, 1.50, 1.80]),
    # Shard density (fragment_size is the real density control; shard_budget is a ceiling)
    # jointly with source_radius, which memory says only pays off jointly with density.
    "density": dict(fragment_size=[0.05, 0.07, 0.09, 0.12],
                    source_radius=[0.0015, 0.005, 0.010]),
    # Sheet layout: more sheets per view, and how far apart they sit.
    "layout": dict(n_per_family=[1, 2, 3], pitch=[0.03, 0.06, 0.09]),
    # Follow-up: pitch turned out to be the strongest lever found, and tighter was better at
    # every sheet count. A tight stack keeps every sheet near the turntable axis, so a
    # shard's stray shadow on the other views lands close to where its own does -- and more
    # sheets give the host greedy more depths to hide the remaining stray light in.
    "layout2": dict(n_per_family=[3, 4, 5, 6], pitch=[0.010, 0.015, 0.020, 0.030]),
    # Where does adding sheets stop paying? Each extra sheet is real perspex and a real
    # weave joint, and inside a fixed 30 cm swept circle a taller stack also FORCES the
    # sheets shorter (the footprint solve trades length for offset), so this axis has to
    # turn over somewhere.
    "layout3": dict(n_per_family=[6, 8, 10, 12], pitch=[0.015, 0.020, 0.025]),
    # Damped pre-compensation. The undamped map oscillates; does damping recover it?
    "precomp": dict(precompensate=[2], precompensate_gain=[0.0, 0.25, 0.5, 0.75, 1.0]),
    # Where to put the three engraved tones. The laser can be driven to any density, so this
    # is a quantiser-design question -- three cut-points on a tone axis -- not a stock list.
    "engrave": dict(engrave_levels=list(engrave.LEVEL_SWEEPS.values())),
    # Does the 4-tone alphabet actually need lamination, given stacking multiplies?
    "stack": dict(max_stack=[1, 2, 3, 4],
                  engrave_levels=[engrave.LEVEL_SWEEPS["wide"],
                                  engrave.LEVEL_SWEEPS["dark_biased"]]),
    # Tone, not silhouette. The reconstructions come out darker than the targets while IoU
    # stays high, because IoU scores only the outline of the mark; this is the axis that
    # SSIM and RMSE actually respond to.
    "tone": dict(intensity_gain=[0.6, 0.75, 0.9, 1.0, 1.15],
                 max_stack=[1, 2, 3]),
    # Band-limiting: stop the solver spending shards on target detail the optics will destroy
    # anyway. One wall pixel is 3 mm; penumbra sigma is ~1.3 px and the min feature projects
    # to ~11 px, so the reproducible band ends well below the target's own bandwidth. Scored
    # against the UNBLURRED target throughout, so any gain here is real.
    "bandlimit": dict(target_blur_px=[0.0, 1.0, 2.0, 3.0, 4.0, 6.0]),
    # Does band-limiting change what the right tone gain is? If the solver stops chasing
    # unreproducible edges it has shards to spare, which may want a different exposure.
    "bandtone": dict(target_blur_px=[0.0, 2.0, 3.0],
                     intensity_gain=[0.85, 0.9, 1.0]),
    # --- six-sheet re-optimisation (arm 30v6) ------------------------------------------
    # Everything below re-asks, at 6 sheets, a question that was answered at 18. The answers
    # are not expected to carry: at 18 the cross-talk had turned constructive (-0.042 mean
    # IoU) and the tuning leaned on it, at 6 it costs +0.017 and is noise again.
    #
    # Pitch was pinned at 20 mm by the footprint solve when six sheets per family had to
    # share a 30 cm swept circle. Two sheets span one gap, not five, so the cap is ~5x
    # looser and the tight-stack argument (keep every sheet near the axis so stray shadows
    # land near their own) is worth re-testing over a range it was never tested on.
    "v6pitch": dict(pitch=[0.02, 0.03, 0.05, 0.07, 0.10]),
    # The tone alphabet is the lever most likely to have moved. Reachable tones are products
    # of the layers a ray crosses, so cutting the stack from 6 sheets per family to 2 cuts
    # the reachable set roughly quadratically -- the darks that 18 sheets reached by
    # multiplying light levels now have to be engraved directly.
    "v6engrave": dict(engrave_levels=list(engrave.LEVEL_SWEEPS.values())),
    # The probe over-covers on every view (fg 0.43 predicted vs 0.36 wanted on front), which
    # is a tone error IoU cannot see. Fewer layers to multiply means each one must be lighter.
    # `v6engrave` confirmed the direction hard: the two lightest alphabets took the top two
    # places and the dark-biased one that won at 18 sheets fell to sixth. This refines around
    # the winners and asks whether the gain wants to come down with them.
    "v6tone": dict(intensity_gain=[0.8, 0.9, 1.0],
                   engrave_levels=[engrave.LEVEL_SWEEPS["shallow"],
                                   engrave.LEVEL_SWEEPS["light_biased"],
                                   (0.82, 0.66, 0.50), (0.72, 0.54, 0.38)]),
    # 767 shards landed on 6 sheets rather than 18, so each sheet is 3x denser. Density was
    # measured flat at 18 sheets, but flat there meant "the stack absorbs it"; here it may not.
    "v6density": dict(fragment_size=[0.05, 0.07, 0.09, 0.12],
                      shard_budget=[180, 260, 400]),
    # Cross-talk is a cost again, so the price put on it at host-selection time should matter
    # more than it did at 18 sheets, where it was measured binary on/off.
    "v6damage": dict(damage_weight=[0.02, 0.10, 0.25, 0.5, 1.0]),
}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", choices=sorted(SWEEPS))
    ap.add_argument("--arm", choices=sorted(ARMS), default="30v2")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank", default="score", choices=["score", "mean_iou", "min_iou", "ssim"])
    a = ap.parse_args(argv)

    base = ARMS[a.arm]
    out = Path(a.out or f"out_pearl3/sweep_{a.sweep}_{a.arm}")
    sw = Sweep(base, out, range(a.seeds))
    baseline = sw.evaluate({})
    print(f"baseline ({a.arm}): score {baseline['score']:.3f}  mean IoU {baseline['mean_iou']:.3f} "
          f"+-{baseline['std_mean_iou']:.3f}  ssim {baseline['ssim']:.3f}  "
          f"cross-talk cost {baseline['crosstalk_cost']:.3f}")
    rows = sw.grid(SWEEPS[a.sweep], f"{a.sweep} on {a.arm}")
    report(rows, baseline, rank=a.rank)


if __name__ == "__main__":
    main()
