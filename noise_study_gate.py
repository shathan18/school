"""Where is the knee? Sweep the perceptual credit gate to find how low noise can go.

noise_study.py isolated ONE lever that cut bad cross-talk without costing duty: judging
"right colour" in CIELAB dE instead of raw-RGB Euclidean distance.

Mechanism (this is the whole story):

  _shard_damage scores a candidate placement as  signed = e_with - e_without  where
  e_without = ||1 - target||^2 is the error of leaving the pixel blank white. On a DARK
  subject pixel e_without is huge, so almost ANY dark shard scores negative -- i.e. earns
  credit -- regardless of its hue. The `match_tol` gate is what is supposed to stop that,
  but in RGB a dark wrong-hue shard sits close to a dark target:

      target (0.10,0.10,0.10) vs shard (0.30,0.05,0.05)  ->  RGB dist 0.21  < 0.30  PASSES
                                                              CIELAB dE   ~35    > 15  BLOCKED

  RGB distance collapses hue differences at low luminance; CIELAB does not. So the RGB gate
  hands "double duty" credit to shards that read as coloured stains on dark regions -- which
  is exactly the noise being complained about.

This sweep finds where tightening dE starts costing genuine duty instead of just noise.
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
REPORT_TOL, REPORT_METRIC = 25.0, "lab"      # fixed ruler for every arm

PAIRS = [
    ("examples/girl_front_nobg.png", "examples/girl_back_nobg.png", "pearl_earring"),
    ("examples/wave_src.jpg", "examples/blue_fuji_v2.png", "wave_fuji"),
]

# (label, kwargs)
ARMS = [("rgb 0.30 (current)", dict(match_tol=0.30, match_metric="rgb", credit_weight=1.0))]
for tol in (30.0, 20.0, 15.0, 12.0, 9.0, 6.0):
    ARMS.append((f"lab dE{tol:g}", dict(match_tol=tol, match_metric="lab", credit_weight=1.0)))
# best-of gate, with credit turned up to buy back any duty the tighter gate costs
for tol in (12.0, 9.0):
    ARMS.append((f"lab dE{tol:g} c2.0",
                 dict(match_tol=tol, match_metric="lab", credit_weight=2.0)))


def _mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else float("nan")


def evaluate(scene, panels, targets, names, seed, kw):
    sl = dataclasses.replace(scene, panels=panels)
    table = build_projection_table(sl)
    renderer = Renderer(sl, table)
    out = decompose.fragment_shards_overlap(
        sl, table, targets, names=names, white_thr=sl.white_threshold,
        max_stack=sl.color_max_stack, seed=seed, damage_weight=0.5, **kw)
    panel_T = C.stack_transmit_lut(names, out[0], out[6])
    acc = _metrics.evaluate_wall_accuracy(targets, renderer.render_color_np(panel_T))
    prim = {p.name: primary_wall_of(sl, table, p) for p in panels}
    good, bad = colour_agreeing_duty(renderer, panel_T, panels, targets, sl.white_threshold,
                                     prim=prim, match_tol=REPORT_TOL,
                                     match_metric=REPORT_METRIC)
    return {"A_ssim": acc["A"]["ssim"], "B_ssim": acc["B"]["ssim"],
            "A_edge": acc["A"]["edge_fidelity"], "B_edge": acc["B"]["edge_fidelity"],
            "good_A": good["A"], "good_B": good["B"],
            "bad_A": bad["A"], "bad_B": bad["B"],
            "panels_used": len({f["panel"] for f in out[2]})}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    scene = load_scene(SCENE)
    names = ["clear"] + C.CMYK
    rows, t0 = [], time.time()

    for a_img, b_img, label in PAIRS:
        print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
        targets = {"A": C.load_color_target(a_img, scene.solve.wall_res,
                                            white_thr=scene.white_threshold),
                   "B": C.load_color_target(b_img, scene.solve.wall_res,
                                            white_thr=scene.white_threshold)}
        panels_by_seed = {s: build_panels_greedy(scene, count=PANEL_COUNT, mode="deliberate",
                                                 K=K_CANDIDATES, targets=targets, seed=s,
                                                 angle_deg_range=ANGLE_RANGE)[0]
                          for s in SEEDS}
        print(f"  {'gate':<20} {'good%':>6} {'bad%':>6} {'g/b':>6} "
              f"{'dbad%':>7} {'dgood%':>7} {'SSIM':>6} {'edge':>6}")
        base = None
        for arm, kw in ARMS:
            per = []
            for seed in SEEDS:
                r = evaluate(scene, panels_by_seed[seed], targets, names, seed, kw)
                r.update(label=label, arm=arm, seed=seed)
                rows.append(r); per.append(r)
            g = _mean(0.5 * (r["good_A"] + r["good_B"]) for r in per)
            b = _mean(0.5 * (r["bad_A"] + r["bad_B"]) for r in per)
            ss = _mean(0.5 * (r["A_ssim"] + r["B_ssim"]) for r in per)
            ed = _mean(0.5 * (r["A_edge"] + r["B_edge"]) for r in per)
            if base is None:
                base = (g, b)
            dg = 100.0 * (g - base[0]) / base[0]
            db = 100.0 * (b - base[1]) / base[1]
            flag = "  <<<" if (db < -8.0 and dg > -8.0) else ""
            print(f"  {arm:<20} {g:6.2f} {b:6.2f} {g / b:6.2f} "
                  f"{db:+7.1f} {dg:+7.1f} {ss:6.3f} {ed:6.3f}{flag}")

    (OUT_DIR / "gate_runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows),
                                             encoding="utf-8")
    print(f"\ndbad%/dgood% are relative to the rgb 0.30 baseline. '<<<' = noise down >8% "
          f"with duty held within 8%.")
    print(f"Wrote {OUT_DIR / 'gate_runs.jsonl'} -- {len(rows)} runs in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
