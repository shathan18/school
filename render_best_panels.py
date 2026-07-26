"""Render preview PNGs + interactive HTML for the best (panel_count, seed) per image
pair from the panel-count sweep. Reads out_panel_sweep/runs.jsonl, picks the winner per
label by composite score, rebuilds that exact panel layout with the same signed-damage
+ credit host selection the sweep used, and writes the standard colour-overlap outputs
into out_panel_sweep/<label>_pc<K>_seed<S>/.

Kept in sync with sweep_panels.py -- ANGLE_RANGE / DAMAGE_WEIGHT / CREDIT_WEIGHT here
must match sweep_panels.py so the previews reproduce the numbers in runs.jsonl.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from shadowart.config.io import load_scene
from shadowart.forward.renderer import Renderer
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.preview.interactive3d import build_interactive
from shadowart.preview.wallview import save_color_comparison
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy, joint_intersection_pct
from shadowart.solve.search import colour_agreeing_duty
from shadowart.targets import color as C
from shadowart import metrics as _metrics

SCENE = "scenes/example.yaml"
SWEEP_DIR = Path("out_panel_sweep")

# MUST match sweep_panels.py
ANGLE_RANGE = (30, 60)
K_CANDIDATES = 16
DAMAGE_WEIGHT = 0.5
CREDIT_WEIGHT = 1.0

# Credit gate -- see out_noise_study/. The default raw-RGB gate (match_tol=0.30) collapses hue
# differences at low luminance, so a wrong-hue DARK shard sits within 0.30 of a dark target and
# earns "double duty" credit while actually reading as a coloured stain. CIELAB dE separates hue
# at low luminance and blocks it. Measured: bad cross-talk -6% to -28% with duty held, and SSIM
# and edge-fidelity both IMPROVE -- it is not a tradeoff.
MATCH_METRIC = "lab"
MATCH_TOL_DEFAULT = 12.0          # arbitrary pairs: little genuine agreement to protect
MATCH_TOL = {"pearl_earring": 20.0}   # palette-compatible pairs: keep the gate loose enough


def main():
    rows = [json.loads(l) for l in (SWEEP_DIR / "runs.jsonl").read_text().splitlines()]
    best = {}
    for r in rows:
        if r["label"] not in best or r["score"] > best[r["label"]]["score"]:
            best[r["label"]] = r
    print("Rendering best config per image pair:")
    for lbl, r in best.items():
        print(f"  {lbl}: pc={r['panel_count']} seed={r['seed']}  "
              f"score={r['score']:.3f}  joint@0.2={r['joint_02']:.1f}%")

    scene = load_scene(SCENE)
    wr = scene.solve.wall_res
    names = ["clear"] + C.CMYK

    for lbl, r in best.items():
        pc, seed = r["panel_count"], r["seed"]
        a_img, b_img = r["target_a"], r["target_b"]
        out = SWEEP_DIR / f"{lbl}_pc{pc}_seed{seed}"
        out.mkdir(exist_ok=True)
        print(f"\n=== {lbl}  pc={pc} seed={seed}  -> {out} ===")

        targets = {"A": C.load_color_target(a_img, wr, white_thr=scene.white_threshold),
                   "B": C.load_color_target(b_img, wr, white_thr=scene.white_threshold)}
        panels, _ = build_panels_greedy(
            scene, count=pc, mode="deliberate", K=K_CANDIDATES,
            targets=targets, seed=seed, angle_deg_range=ANGLE_RANGE,
        )
        scene_layout = dataclasses.replace(scene, panels=panels)
        table = build_projection_table(scene_layout)
        renderer = Renderer(scene_layout, table)

        stack_colorid, opacity, fragments, resolved, stack_depths, budget_stats, stack_intensity = \
            decompose.fragment_shards_overlap(
                scene_layout, table, targets, names=names,
                white_thr=scene_layout.white_threshold,
                max_stack=scene_layout.color_max_stack, seed=seed,
                damage_weight=DAMAGE_WEIGHT, credit_weight=CREDIT_WEIGHT,
                match_metric=MATCH_METRIC,
                match_tol=MATCH_TOL.get(lbl, MATCH_TOL_DEFAULT))
        panel_T = C.stack_transmit_lut(names, stack_colorid, stack_intensity)
        pred_rgb = renderer.render_color_np(panel_T)

        acc = _metrics.evaluate_wall_accuracy(targets, pred_rgb)
        joint = joint_intersection_pct(fragments, table, panels)
        # Honest cross-talk split (corrections_note.md 3): `joint` above is colour-BLIND and
        # reads ~100% for every layout, so it cannot see noise. good = stray shadow that landed
        # in the colour that wall wants; bad = wrong-colour contamination.
        prim = {p.name: primary_wall_of(scene_layout, table, p) for p in panels}
        duty, bleed = colour_agreeing_duty(renderer, panel_T, panels, targets,
                                           scene_layout.white_threshold, prim=prim,
                                           match_tol=25.0, match_metric="lab")
        print(f"  double duty (colour-agreeing)  A={duty['A']:.1f}%  B={duty['B']:.1f}%")
        print(f"  noise (wrong-colour bleed)     A={bleed['A']:.1f}%  B={bleed['B']:.1f}%")
        print(_metrics.format_accuracy_report(acc))
        print(f"  double-duty (shard area with balanced two-wall contribution): "
              f"@0.1={joint[0.1]:.1f}%  @0.2={joint[0.2]:.1f}%  @0.3={joint[0.3]:.1f}%")

        save_color_comparison(targets, pred_rgb, out / "preview_final.png")
        np.save(out / "opacity.npy", opacity)
        np.save(out / "stack_colorid.npy", stack_colorid)
        for fam in ("A", "B"):
            C.save_cmyk_channels(targets[fam], out / "cmyk_channels", fam)

        stack_pieces = decompose.panel_stack_pieces(scene_layout, stack_colorid, names)
        poly_channel = {id(poly): ch for items in stack_pieces.values() for poly, ch, _s in items}
        flat_pieces = {name: [poly for poly, _ch, _s in items]
                       for name, items in stack_pieces.items()}
        stack_color_of = lambda panel, poly: tuple(
            C.display_rgb(poly_channel.get(id(poly), "clear")))
        build_interactive(scene_layout, table, opacity, None,
                          out / "scene_interactive.html", rays=40, auto_open=False,
                          wall_rgb=pred_rgb, pieces=flat_pieces, color_of=stack_color_of)
        print(f"  wrote preview_final.png, scene_interactive.html, opacity.npy, "
              f"stack_colorid.npy, cmyk_channels/")


if __name__ == "__main__":
    main()
