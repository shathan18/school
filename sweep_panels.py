"""Sweep panel count on several image pairs -- diagonal-only, INTERSECTION-FIRST.

Differs from the previous sweep in three ways:
  1. Angles restricted to (30, 60) degrees so every panel is a real diagonal that
     projects meaningfully onto BOTH walls (5-85 degrees still gave diagonals but
     included near-axis-aligned angles whose cross-wall projection is tiny).
  2. Shard host selection uses SIGNED damage with credit:
        damage_weight = 0.5    (avoid bad cross-talk)
        credit_weight = 1.0    (REWARD shards that also help the other wall's image)
     Per report_team.md this puts us in the 32.7% good-double-duty / 7% bad-noise
     regime -- consistent with "must have intersection, tolerate some noise".
  3. Records joint_intersection_pct (at thresholds 0.1 / 0.2 / 0.3) alongside the
     per-wall RMSE / SSIM / edge-fidelity, and ranks winners by
        score = mean_SSIM + 0.5 * mean_edge + 1.0 * joint@0.2 / 100
     so a layout with more genuine double-duty wins ties against a slightly
     sharper but purely one-wall layout.

Same panel-count grid, same image pairs, same 2 seeds as before.
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import time
from pathlib import Path

from shadowart.config.io import load_scene
from shadowart.forward.renderer import Renderer
from shadowart.geometry.projection import build_projection_table
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy, joint_intersection_pct
from shadowart.targets import color as C
from shadowart import metrics as _metrics

# --------------------------------------------------------------------- CONFIG
SCENE = "scenes/example.yaml"
OUT_DIR = Path("out_panel_sweep")

PANEL_COUNTS = [4, 6, 8, 10, 12, 14, 18, 22]
SEEDS = [1, 2]

# Diagonal-only: 30-60 degrees. is_diagonal(tol=5) fires at 5-85 already but that
# still allowed near-axis angles whose cross-wall projection area is tiny -- a shard
# on a 10-deg panel casts mostly on Wall B and barely on Wall A. 30-60 forces the
# projected footprint on both walls to be comparable, which is what "intersection"
# needs geometrically.
ANGLE_RANGE = (30, 60)
K_CANDIDATES = 16

# Signed-damage + credit host selection: rewards shards whose stray shadow lands on
# the OTHER wall's real image content (double-duty) instead of just avoiding it.
# 0.5 / 1.0 is a moderately aggressive credit -- more double-duty than 0.5/0.5, less
# noise than 0.5/2.0. See report_team.md and out_thickness_test/damage_test.py.
DAMAGE_WEIGHT = 0.5
CREDIT_WEIGHT = 1.0

# Weight on joint_intersection_pct@0.2 in the composite ranking score (as a fraction,
# so 30% double-duty adds 0.30 to score). Set to 1.0 to make intersection a peer of
# mean-SSIM; 0.0 to score fidelity only.
INTERSECT_WEIGHT = 1.0

PAIRS = [
    ("examples/girl_front_nobg.png", "examples/girl_back_nobg.png", "pearl_earring"),
    ("examples/wave_src.jpg", "examples/blue_fuji_v2.png", "wave_fuji"),
    ("examples/apples.jpg", "examples/breakfast.jpg", "apples_breakfast"),
]


# --------------------------------------------------------------------- HELPERS
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else float("nan")


def evaluate(scene, panels, targets, names, seed):
    """Run the colour-overlap pipeline with signed-damage + credit host selection,
    then compute per-wall accuracy AND joint_intersection_pct."""
    scene_layout = dataclasses.replace(scene, panels=panels)
    table = build_projection_table(scene_layout)
    renderer = Renderer(scene_layout, table)
    stack_colorid, opacity, fragments, resolved, stack_depths, budget_stats, stack_intensity = \
        decompose.fragment_shards_overlap(
            scene_layout, table, targets, names=names,
            white_thr=scene_layout.white_threshold,
            max_stack=scene_layout.color_max_stack, seed=seed,
            damage_weight=DAMAGE_WEIGHT, credit_weight=CREDIT_WEIGHT)
    panel_T = C.stack_transmit_lut(names, stack_colorid, stack_intensity)
    pred_rgb = renderer.render_color_np(panel_T)
    acc = _metrics.evaluate_wall_accuracy(targets, pred_rgb)
    joint = joint_intersection_pct(fragments, table, panels)      # {0.1: %, 0.2: %, 0.3: %}
    return {
        "accuracy": acc,
        "joint": joint,
        "shard_regions": {w: budget_stats.get(w, {}).get("achieved", 0) for w in scene.walls},
        "n_shards": len(fragments),
        "n_diagonals": sum(1 for p in panels if p.is_diagonal()),
    }


def _score(result):
    """Composite: mean-SSIM + 0.5*mean-edge + INTERSECT_WEIGHT * joint@0.2 fraction."""
    acc = result["accuracy"]
    ssim = (acc["A"]["ssim"] + acc["B"]["ssim"]) / 2.0
    edge = (acc["A"]["edge_fidelity"] + acc["B"]["edge_fidelity"]) / 2.0
    j = result["joint"][0.2] / 100.0
    return ssim + 0.5 * edge + INTERSECT_WEIGHT * j


def _fmt(x, spec=".3f"):
    return format(x, spec) if x == x else " nan"


# --------------------------------------------------------------------- MAIN
def main():
    OUT_DIR.mkdir(exist_ok=True)
    scene = load_scene(SCENE)
    wr = scene.solve.wall_res
    names = ["clear"] + C.CMYK

    all_rows = []
    aggregate = {}
    total_runs = len(PAIRS) * len(PANEL_COUNTS) * len(SEEDS)
    run_i = 0
    t_start = time.time()

    for a_img, b_img, label in PAIRS:
        print(f"\n=== {label}  A={a_img}  B={b_img} ===")
        targets = {
            "A": C.load_color_target(a_img, wr, white_thr=scene.white_threshold),
            "B": C.load_color_target(b_img, wr, white_thr=scene.white_threshold),
        }

        for pc in PANEL_COUNTS:
            per_seed = []
            for seed in SEEDS:
                run_i += 1
                t0 = time.time()
                panels, _ = build_panels_greedy(
                    scene, count=pc, mode="deliberate", K=K_CANDIDATES,
                    targets=targets, seed=seed, angle_deg_range=ANGLE_RANGE,
                )
                if len(panels) < pc:
                    print(f"  [{run_i}/{total_runs}] pc={pc} seed={seed}: only "
                          f"{len(panels)}/{pc} panels placeable -- skipping")
                    continue
                result = evaluate(scene, panels, targets, names, seed)
                dt = time.time() - t0
                acc = result["accuracy"]
                j = result["joint"]
                score = _score(result)
                row = {
                    "label": label, "target_a": a_img, "target_b": b_img,
                    "panel_count": pc, "n_diagonals": result["n_diagonals"],
                    "seed": seed, "score": score,
                    "A_rmse": acc["A"]["rmse"], "A_ssim": acc["A"]["ssim"],
                    "A_edge": acc["A"]["edge_fidelity"], "A_psnr": acc["A"]["psnr_db"],
                    "B_rmse": acc["B"]["rmse"], "B_ssim": acc["B"]["ssim"],
                    "B_edge": acc["B"]["edge_fidelity"], "B_psnr": acc["B"]["psnr_db"],
                    "joint_01": j[0.1], "joint_02": j[0.2], "joint_03": j[0.3],
                    "shard_regions": result["shard_regions"],
                    "n_shards": result["n_shards"],
                    "elapsed_s": dt,
                }
                all_rows.append(row)
                per_seed.append(row)
                print(f"  [{run_i}/{total_runs}] pc={pc:2d} seed={seed} | "
                      f"A ssim={acc['A']['ssim']:.3f} edge={acc['A']['edge_fidelity']:.3f} | "
                      f"B ssim={acc['B']['ssim']:.3f} edge={acc['B']['edge_fidelity']:.3f} | "
                      f"joint 0.1/0.2/0.3={j[0.1]:4.1f}/{j[0.2]:4.1f}/{j[0.3]:4.1f}% | "
                      f"score={score:.3f} ({dt:.1f}s)")
            if per_seed:
                aggregate[(label, pc)] = {
                    "label": label, "panel_count": pc, "n_seeds": len(per_seed),
                    "A_ssim": _mean(r["A_ssim"] for r in per_seed),
                    "A_edge": _mean(r["A_edge"] for r in per_seed),
                    "A_rmse": _mean(r["A_rmse"] for r in per_seed),
                    "B_ssim": _mean(r["B_ssim"] for r in per_seed),
                    "B_edge": _mean(r["B_edge"] for r in per_seed),
                    "B_rmse": _mean(r["B_rmse"] for r in per_seed),
                    "joint_01": _mean(r["joint_01"] for r in per_seed),
                    "joint_02": _mean(r["joint_02"] for r in per_seed),
                    "joint_03": _mean(r["joint_03"] for r in per_seed),
                    "score": _mean(r["score"] for r in per_seed),
                }

    (OUT_DIR / "runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in all_rows), encoding="utf-8")
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "config": {"scene": SCENE, "panel_counts": PANEL_COUNTS, "seeds": SEEDS,
                       "K_candidates": K_CANDIDATES, "angle_deg_range": ANGLE_RANGE,
                       "damage_weight": DAMAGE_WEIGHT, "credit_weight": CREDIT_WEIGHT,
                       "intersect_weight": INTERSECT_WEIGHT},
            "pairs": [{"label": lbl, "a": a, "b": b} for a, b, lbl in PAIRS],
            "aggregate": [v for v in aggregate.values()],
        }, fh, indent=2)

    print("\n" + "=" * 92)
    print("SUMMARY  score = mean_SSIM + 0.5*mean_edge + "
          f"{INTERSECT_WEIGHT:.1f}*(joint@0.2/100)  (mean across seeds)")
    print("=" * 92)
    for _, _, label in PAIRS:
        rows = [aggregate[(lbl, pc)] for (lbl, pc) in aggregate if lbl == label]
        if not rows:
            continue
        rows.sort(key=lambda r: r["panel_count"])
        print(f"\n{label}")
        print(f"  {'pc':>3} | {'score':>5} | "
              f"{'A_ssim':>6} {'A_edge':>6} | {'B_ssim':>6} {'B_edge':>6} | "
              f"{'j@0.1':>6} {'j@0.2':>6} {'j@0.3':>6}")
        best = max(rows, key=lambda r: r["score"])
        for r in rows:
            mark = "  <-- best" if r is best else ""
            print(f"  {r['panel_count']:>3d} | {_fmt(r['score']):>5} | "
                  f"{_fmt(r['A_ssim']):>6} {_fmt(r['A_edge']):>6} | "
                  f"{_fmt(r['B_ssim']):>6} {_fmt(r['B_edge']):>6} | "
                  f"{r['joint_01']:>5.1f}% {r['joint_02']:>5.1f}% {r['joint_03']:>5.1f}%"
                  f"{mark}")

    print("\n" + "=" * 92)
    print("BEST panel count per image  (composite score, mean across seeds)")
    print("=" * 92)
    for _, _, label in PAIRS:
        rows = [aggregate[(lbl, pc)] for (lbl, pc) in aggregate if lbl == label]
        if not rows:
            continue
        best = max(rows, key=lambda r: r["score"])
        print(f"  {label:20s}  pc={best['panel_count']:2d}   score={best['score']:.3f}   "
              f"joint@0.2={best['joint_02']:.1f}%   "
              f"A_ssim={best['A_ssim']:.3f}  B_ssim={best['B_ssim']:.3f}")

    print(f"\nWrote {OUT_DIR/'runs.jsonl'} and {OUT_DIR/'summary.json'}")
    print(f"Total wallclock: {time.time() - t_start:.1f}s over {len(all_rows)} runs")


if __name__ == "__main__":
    main()
