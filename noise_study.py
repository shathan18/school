"""Can we cut BAD cross-talk (noise) without losing GOOD cross-talk (double duty)?

Framing follows corrections_note.md 3: the colour-BLIND `joint_intersection_pct` (which
reads ~100% in out_panel_sweep) counts every stray shadow that merely LANDS on the other
image's subject. The honest split is `search.colour_agreeing_duty`:

    good = stray shadow darkened a subject pixel AND arrived in ~the colour that wall wants
    bad  = stray shadow darkened a subject pixel in the WRONG colour  <-- this is the noise

So the question is: which knobs lower `bad` while holding `good`?

Levers swept (all already exist in decompose.fragment_shards_overlap; none are new code):

  match_metric  rgb -> lab   Judge "right colour" in CIELAB dE instead of raw-RGB Euclidean
                             distance. RGB distance is not perceptually uniform, so on dark /
                             near-neutral targets it lets visibly-wrong colours pass the credit
                             gate. Should strip credit from contaminating shards -> less bad.
  match_tol                  How close the transmitted colour must be to earn credit. Tighter
                             = fewer wrong-colour shards rewarded = less bad (but risks
                             starving good too).
  outline_protect_weight     Uses outline_masks (decompose.outline_map) to make the OTHER
                             wall's defining contour expensive to land on. _shard_damage is
                             edge-blind -- a stray shadow on a dark contour steals ~no light so
                             it scores harmless, which is precisely what lets contamination
                             pile onto the line carrying the image. This should be the
                             highest-leverage noise knob.
  credit_weight              How hard genuine double duty is rewarded.

Reports per arm: good%, bad%, good/bad ratio, SSIM, edge-fidelity, plus the colour-blind
joint% for reference (to show it stays ~100 and is therefore uninformative).
"""
from __future__ import annotations

import dataclasses
import json
import statistics
import time
from pathlib import Path

import numpy as np

from shadowart.config.io import load_scene
from shadowart.forward.renderer import Renderer
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy, joint_intersection_pct
from shadowart.solve.search import colour_agreeing_duty
from shadowart.targets import color as C
from shadowart import metrics as _metrics

# --------------------------------------------------------------------- CONFIG
SCENE = "scenes/example.yaml"
OUT_DIR = Path("out_noise_study")

# Fixed geometry across every arm so ONLY the assignment/credit knobs vary.
PANEL_COUNT = 14
ANGLE_RANGE = (30, 60)        # diagonal-only, same as the panel-count sweep
K_CANDIDATES = 16
SEEDS = [1, 2, 3]

PAIRS = [
    # Palette-COMPATIBLE pair -- corrections_note.md 4 says this is where genuine double
    # duty actually exists (~24%), so it is the pair where "keep good, cut bad" is a real
    # question rather than a fight over ~1% of signal.
    ("examples/girl_front_nobg.png", "examples/girl_back_nobg.png", "pearl_earring"),
    # Arbitrary pair -- honest duty is near zero here; included to confirm the knobs do not
    # manufacture agreement that isn't in the images.
    ("examples/wave_src.jpg", "examples/blue_fuji_v2.png", "wave_fuji"),
]

# name -> kwargs forwarded to fragment_shards_overlap. damage_weight fixed at 0.5 throughout.
ARMS = {
    # --- controls ---------------------------------------------------------
    "random":            dict(damage_weight=0.0, credit_weight=None),
    "harm_only":         dict(damage_weight=0.5, credit_weight=None),
    "baseline_rgb_c1.0": dict(damage_weight=0.5, credit_weight=1.0,
                              match_tol=0.30, match_metric="rgb"),
    # --- lever 1: perceptual colour gate ---------------------------------
    "lab_dE25":          dict(damage_weight=0.5, credit_weight=1.0,
                              match_tol=25.0, match_metric="lab"),
    "lab_dE15":          dict(damage_weight=0.5, credit_weight=1.0,
                              match_tol=15.0, match_metric="lab"),
    # --- lever 2: tighter RGB gate ---------------------------------------
    "rgb_tol0.20":       dict(damage_weight=0.5, credit_weight=1.0,
                              match_tol=0.20, match_metric="rgb"),
    "rgb_tol0.12":       dict(damage_weight=0.5, credit_weight=1.0,
                              match_tol=0.12, match_metric="rgb"),
    # --- lever 3: outline protection (needs outline_masks, added at runtime)
    "outline_pw1.0":     dict(damage_weight=0.5, credit_weight=1.0,
                              match_tol=0.30, match_metric="rgb",
                              outline_protect_weight=1.0),
    "outline_pw3.0":     dict(damage_weight=0.5, credit_weight=1.0,
                              match_tol=0.30, match_metric="rgb",
                              outline_protect_weight=3.0),
    # --- lever 4: lower credit -------------------------------------------
    "credit0.5":         dict(damage_weight=0.5, credit_weight=0.5,
                              match_tol=0.30, match_metric="rgb"),
    # --- combined: perceptual gate + outline guard ------------------------
    "lab_dE15+outline3": dict(damage_weight=0.5, credit_weight=1.0,
                              match_tol=15.0, match_metric="lab",
                              outline_protect_weight=3.0),
    "lab_dE15+outline3+c2": dict(damage_weight=0.5, credit_weight=2.0,
                                 match_tol=15.0, match_metric="lab",
                                 outline_protect_weight=3.0),
}

# Metric gate for reporting good/bad. Kept FIXED across arms (perceptual dE 25) so every arm
# is judged by the same ruler even when its internal optimiser gate differs -- otherwise an
# arm could "win" merely by grading itself more leniently.
REPORT_TOL = 25.0
REPORT_METRIC = "lab"


def _mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else float("nan")


def evaluate(scene, panels, targets, names, seed, kwargs, outline_masks):
    scene_layout = dataclasses.replace(scene, panels=panels)
    table = build_projection_table(scene_layout)
    renderer = Renderer(scene_layout, table)

    kw = dict(kwargs)
    if kw.get("outline_protect_weight"):
        kw["outline_masks"] = outline_masks

    stack_colorid, opacity, fragments, resolved, stack_depths, budget_stats, stack_intensity = \
        decompose.fragment_shards_overlap(
            scene_layout, table, targets, names=names,
            white_thr=scene_layout.white_threshold,
            max_stack=scene_layout.color_max_stack, seed=seed, **kw)
    panel_T = C.stack_transmit_lut(names, stack_colorid, stack_intensity)
    pred_rgb = renderer.render_color_np(panel_T)

    acc = _metrics.evaluate_wall_accuracy(targets, pred_rgb)
    prim = {p.name: primary_wall_of(scene_layout, table, p) for p in panels}
    good, bad = colour_agreeing_duty(
        renderer, panel_T, panels, targets, scene_layout.white_threshold,
        prim=prim, match_tol=REPORT_TOL, match_metric=REPORT_METRIC)
    joint = joint_intersection_pct(fragments, table, panels)
    n_used = len({f["panel"] for f in fragments})
    return {
        "A_ssim": acc["A"]["ssim"], "A_edge": acc["A"]["edge_fidelity"],
        "A_rmse": acc["A"]["rmse"],
        "B_ssim": acc["B"]["ssim"], "B_edge": acc["B"]["edge_fidelity"],
        "B_rmse": acc["B"]["rmse"],
        "good_A": good["A"], "good_B": good["B"],
        "bad_A": bad["A"], "bad_B": bad["B"],
        "joint_02": joint[0.2],
        "panels_used": n_used, "n_shards": len(fragments),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    scene = load_scene(SCENE)
    wr = scene.solve.wall_res
    names = ["clear"] + C.CMYK

    rows = []
    t_start = time.time()
    total = len(PAIRS) * len(ARMS) * len(SEEDS)
    i = 0

    for a_img, b_img, label in PAIRS:
        print(f"\n{'=' * 96}\n{label}   A={a_img}   B={b_img}\n{'=' * 96}")
        targets = {"A": C.load_color_target(a_img, wr, white_thr=scene.white_threshold),
                   "B": C.load_color_target(b_img, wr, white_thr=scene.white_threshold)}
        # Outline maps for lever 3: high on each wall image's defining contour.
        outline_masks = {
            w: decompose.outline_map(targets[w],
                                     C.subject_mask(targets[w], scene.white_threshold))
            for w in ("A", "B")
        }

        # Panel geometry is IDENTICAL across arms for a given seed (built before the arm
        # loop) so the only thing changing is shard->panel assignment.
        panels_by_seed = {}
        for seed in SEEDS:
            p, _ = build_panels_greedy(scene, count=PANEL_COUNT, mode="deliberate",
                                       K=K_CANDIDATES, targets=targets, seed=seed,
                                       angle_deg_range=ANGLE_RANGE)
            panels_by_seed[seed] = p

        for arm, kwargs in ARMS.items():
            per_seed = []
            for seed in SEEDS:
                i += 1
                t0 = time.time()
                r = evaluate(scene, panels_by_seed[seed], targets, names, seed,
                             kwargs, outline_masks)
                r.update(label=label, arm=arm, seed=seed, elapsed_s=time.time() - t0)
                rows.append(r)
                per_seed.append(r)
            g = _mean(0.5 * (r["good_A"] + r["good_B"]) for r in per_seed)
            b = _mean(0.5 * (r["bad_A"] + r["bad_B"]) for r in per_seed)
            ss = _mean(0.5 * (r["A_ssim"] + r["B_ssim"]) for r in per_seed)
            ed = _mean(0.5 * (r["A_edge"] + r["B_edge"]) for r in per_seed)
            jt = _mean(r["joint_02"] for r in per_seed)
            ratio = g / b if b > 1e-9 else float("inf")
            print(f"  [{i:3d}/{total}] {arm:24s} good={g:5.2f}%  bad={b:5.2f}%  "
                  f"g/b={ratio:6.2f}  SSIM={ss:.3f}  edge={ed:.3f}  "
                  f"joint(blind)={jt:5.1f}%")

    (OUT_DIR / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows),
                                        encoding="utf-8")

    # ------------------------------------------------------------ REPORT
    print(f"\n{'=' * 96}")
    print("NOISE STUDY  --  good = colour-agreeing double duty (keep it) ; "
          "bad = wrong-colour bleed (the noise)")
    print(f"judged at a FIXED gate for every arm: {REPORT_METRIC} dE < {REPORT_TOL}")
    print(f"{'=' * 96}")

    for _, _, label in PAIRS:
        sub = [r for r in rows if r["label"] == label]
        if not sub:
            continue
        print(f"\n{label}")
        print(f"  {'arm':24s} {'good%':>6} {'bad%':>6} {'g/b':>6} | "
              f"{'SSIM':>5} {'edge':>5} | {'panels':>6} {'joint':>6}")
        base = None
        stats = []
        for arm in ARMS:
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            s = {
                "arm": arm,
                "good": _mean(0.5 * (r["good_A"] + r["good_B"]) for r in a),
                "bad": _mean(0.5 * (r["bad_A"] + r["bad_B"]) for r in a),
                "ssim": _mean(0.5 * (r["A_ssim"] + r["B_ssim"]) for r in a),
                "edge": _mean(0.5 * (r["A_edge"] + r["B_edge"]) for r in a),
                "panels": _mean(r["panels_used"] for r in a),
                "joint": _mean(r["joint_02"] for r in a),
            }
            s["ratio"] = s["good"] / s["bad"] if s["bad"] > 1e-9 else float("inf")
            stats.append(s)
            if arm == "baseline_rgb_c1.0":
                base = s
        for s in stats:
            tag = ""
            if base and s["arm"] != "baseline_rgb_c1.0":
                dg = s["good"] - base["good"]
                db = s["bad"] - base["bad"]
                if db < -0.01 and dg > -0.01:
                    tag = "  *** less noise, duty held"
                elif db < -0.01:
                    tag = f"  (noise {db:+.2f}, duty {dg:+.2f})"
            print(f"  {s['arm']:24s} {s['good']:6.2f} {s['bad']:6.2f} {s['ratio']:6.2f} | "
                  f"{s['ssim']:.3f} {s['edge']:.3f} | {s['panels']:6.1f} {s['joint']:5.1f}%{tag}")

        if base:
            cands = [s for s in stats
                     if s["arm"] not in ("random", "harm_only", "baseline_rgb_c1.0")]
            if cands:
                best = max(cands, key=lambda s: s["ratio"])
                print(f"\n  best good/bad ratio: {best['arm']}  "
                      f"({best['ratio']:.2f} vs baseline {base['ratio']:.2f})  "
                      f"good {base['good']:.2f}->{best['good']:.2f}%  "
                      f"bad {base['bad']:.2f}->{best['bad']:.2f}%")

    print(f"\nWrote {OUT_DIR / 'runs.jsonl'}")
    print(f"Total wallclock: {time.time() - t_start:.1f}s over {len(rows)} runs")


if __name__ == "__main__":
    main()
