"""Structural noise lever: stop PLACING panels that splatter onto the other wall's background.

noise_study.py showed the assignment-time knobs (colour gate, outline protection, credit
weight) only shave ~20% off bad cross-talk. That is because they all act AFTER the panel set
is fixed: they choose which panel hosts each shard, but if every available panel throws its
stray shadow onto the other wall's background, there is no good choice left to make.

`panel_search._coverage_score` already carries an unused subject-aware term:

    spill_frac = spill / (spill + gain)          # fraction of footprint on BACKGROUND
    score      = gain * (1 - spill_frac) ** spill_weight

with `spill_weight=0.0` (the default everywhere) being exactly spill-blind. Raising it makes
the greedy prefer panels whose footprint lands on SUBJECT on both walls -- which is the same
thing as saying "a stray shadow from this panel has a chance of being useful" -- before any
shard is ever assigned.

This study crosses that placement lever with the assignment gate that won in noise_study.py
(perceptual CIELAB dE < 15), to see whether the two compose.
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import time
from pathlib import Path

from shadowart.config.io import load_scene
from shadowart.forward.renderer import Renderer
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.solve.search import colour_agreeing_duty
from shadowart.targets import color as C
from shadowart import metrics as _metrics

SCENE = "scenes/example.yaml"
OUT_DIR = Path("out_noise_study")
PANEL_COUNT = 14
ANGLE_RANGE = (30, 60)
K_CANDIDATES = 16
SEEDS = [1, 2, 3]
REPORT_TOL, REPORT_METRIC = 25.0, "lab"

PAIRS = [
    ("examples/girl_front_nobg.png", "examples/girl_back_nobg.png", "pearl_earring"),
    ("examples/wave_src.jpg", "examples/blue_fuji_v2.png", "wave_fuji"),
]

SPILL_WEIGHTS = [0.0, 1.0, 2.0, 4.0]
# assignment arms: baseline RGB gate vs the perceptual gate that won noise_study.py
GATES = {
    "rgb0.30": dict(match_tol=0.30, match_metric="rgb"),
    "labdE15": dict(match_tol=15.0, match_metric="lab"),
}


def _mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else float("nan")


def evaluate(scene, panels, targets, names, seed, gate):
    scene_layout = dataclasses.replace(scene, panels=panels)
    table = build_projection_table(scene_layout)
    renderer = Renderer(scene_layout, table)
    out = decompose.fragment_shards_overlap(
        scene_layout, table, targets, names=names,
        white_thr=scene_layout.white_threshold, max_stack=scene_layout.color_max_stack,
        seed=seed, damage_weight=0.5, credit_weight=1.0, **gate)
    stack_colorid, opacity, fragments = out[0], out[1], out[2]
    stack_intensity = out[6]
    panel_T = C.stack_transmit_lut(names, stack_colorid, stack_intensity)
    pred_rgb = renderer.render_color_np(panel_T)
    acc = _metrics.evaluate_wall_accuracy(targets, pred_rgb)
    prim = {p.name: primary_wall_of(scene_layout, table, p) for p in panels}
    good, bad = colour_agreeing_duty(renderer, panel_T, panels, targets,
                                     scene_layout.white_threshold, prim=prim,
                                     match_tol=REPORT_TOL, match_metric=REPORT_METRIC)
    return {"A_ssim": acc["A"]["ssim"], "B_ssim": acc["B"]["ssim"],
            "A_edge": acc["A"]["edge_fidelity"], "B_edge": acc["B"]["edge_fidelity"],
            "good_A": good["A"], "good_B": good["B"],
            "bad_A": bad["A"], "bad_B": bad["B"],
            "panels_used": len({f["panel"] for f in fragments}),
            "n_shards": len(fragments)}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    scene = load_scene(SCENE)
    wr = scene.solve.wall_res
    names = ["clear"] + C.CMYK
    rows = []
    t0 = time.time()

    for a_img, b_img, label in PAIRS:
        print(f"\n{'=' * 92}\n{label}\n{'=' * 92}")
        targets = {"A": C.load_color_target(a_img, wr, white_thr=scene.white_threshold),
                   "B": C.load_color_target(b_img, wr, white_thr=scene.white_threshold)}
        print(f"  {'placement':>10} {'gate':>8} {'good%':>6} {'bad%':>6} {'g/b':>6} "
              f"{'SSIM':>6} {'edge':>6} {'panels':>7}")
        for sw in SPILL_WEIGHTS:
            panels_by_seed = {}
            for seed in SEEDS:
                p, _ = build_panels_greedy(scene, count=PANEL_COUNT, mode="deliberate",
                                           K=K_CANDIDATES, targets=targets, seed=seed,
                                           angle_deg_range=ANGLE_RANGE, spill_weight=sw)
                panels_by_seed[seed] = p
            for gname, gate in GATES.items():
                per = []
                for seed in SEEDS:
                    r = evaluate(scene, panels_by_seed[seed], targets, names, seed, gate)
                    r.update(label=label, spill_weight=sw, gate=gname, seed=seed)
                    rows.append(r); per.append(r)
                g = _mean(0.5 * (r["good_A"] + r["good_B"]) for r in per)
                b = _mean(0.5 * (r["bad_A"] + r["bad_B"]) for r in per)
                ss = _mean(0.5 * (r["A_ssim"] + r["B_ssim"]) for r in per)
                ed = _mean(0.5 * (r["A_edge"] + r["B_edge"]) for r in per)
                pu = _mean(r["panels_used"] for r in per)
                ratio = g / b if b > 1e-9 else float("inf")
                print(f"  spill={sw:<4.1f} {gname:>8} {g:6.2f} {b:6.2f} {ratio:6.2f} "
                      f"{ss:6.3f} {ed:6.3f} {pu:7.1f}")

    (OUT_DIR / "spill_runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows),
                                              encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'spill_runs.jsonl'} -- {len(rows)} runs in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
