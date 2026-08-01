"""Diagnostic: is the projection REAL, and do panels genuinely serve BOTH images?

Three questions, answered by measurement rather than by reading the docstrings:

1. IS THE PROJECTION REAL?
   Ablation. Re-render the scene with exactly one panel made fully clear, and measure how
   much EACH wall changes. If a panel's removal changes both wall A and wall B, that panel
   is physically casting light through to both -- there is no per-family shortcut anywhere
   in the pipeline. This cannot be faked by bookkeeping: it is the actual forward renderer
   (`Renderer.render_color_np`, which warps every panel through its own projective
   homography for every wall) being run twice and differenced.

2. HOW MANY PANELS SERVE BOTH IMAGES AT ONCE?
   `wall_coverage_area` gives each panel's projected footprint on each wall. A panel that is
   nearly edge-on to a wall collapses to a sliver there. We report the SHARED RATIO
   min(areaA, areaB) / max(areaA, areaB): 1.0 = equally face-on to both walls (a true
   double-duty plane), ~0 = serves one wall only. Combined with the ablation delta, this
   says how many planes are really doing double duty.

3. IS THE OVERLAP GENUINE, OR JUST CONTAMINATION?
   `colour_agreeing_duty` splits cross-wall bleed into the half that arrived in a tone the
   other wall actually wanted (good) and the half that is wrong-tone contamination (bad).
   Per shadowart-noise.md, `joint_intersection_pct` is colour-blind and reads ~100% for
   everything -- it is printed here for context only and must never be used to rank.

NOTE ON GRAYSCALE. Earlier I flagged that grayscale discards hue and therefore forfeits the
colour-compatibility result. That flag was half wrong, and this script is how to settle it:
with the `noir` palette BOTH walls want the same neutral greys, so a dark shard is the
correct tone on either wall. Tonal compatibility should be near-maximal. Measured, not
assumed.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from shadowart.config.io import load_scene
from shadowart.forward.renderer import Renderer
from shadowart.geometry.projection import (build_projection_table, primary_wall_of,
                                           wall_coverage_area)
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy, joint_intersection_pct
from shadowart.solve.search import colour_agreeing_duty
from shadowart.targets import color as C

import face_render300 as FR

OUT = Path("out_faces_hc/duty")

# A panel counts as serving a wall at all if its projected footprint there is at least this
# fraction of its footprint on its better wall.
SHARED_KNEE = 0.25
# Ablation: a panel "contributes" to a wall if clearing it moves that wall's mean RGB by
# more than this. 0.002 is ~0.5/255 -- comfortably above float noise, well below visible.
ABLATE_KNEE = 0.002


def diagnose(label: str, ka: str, kb: str, cands: dict, seed: int,
             angle_range=FR.ANGLE_RANGE) -> dict:
    scene = load_scene(FR.SCENE)
    scene = dataclasses.replace(scene, color_palette=C.PALETTES["noir"])
    names = C.palette_names(scene.color_palette)
    wr = scene.solve.wall_res
    targets = {
        "A": C.load_color_target(str(FR.TGT / f"{ka}.png"), wr, white_thr=scene.white_threshold),
        "B": C.load_color_target(str(FR.TGT / f"{kb}.png"), wr, white_thr=scene.white_threshold),
    }
    panels, _ = build_panels_greedy(scene, count=FR.PANEL_COUNT, mode="deliberate",
                                    K=FR.K_CANDIDATES, targets=targets, seed=seed,
                                    angle_deg_range=angle_range)
    layout = dataclasses.replace(scene, panels=panels)
    table = build_projection_table(layout)
    renderer = Renderer(layout, table)

    sc, opacity, fragments, resolved, sd, bs, si = decompose.fragment_shards_overlap(
        layout, table, targets, names=names, white_thr=layout.white_threshold,
        max_stack=layout.color_max_stack, seed=seed, shard_budget=FR.SHARD_BUDGET_PER_WALL,
        damage_weight=FR.DAMAGE_WEIGHT, credit_weight=FR.CREDIT_WEIGHT,
        match_metric=FR.MATCH_METRIC, match_tol=FR.MATCH_TOL)
    panel_T = C.stack_transmit_lut(names, sc, si)
    base = renderer.render_color_np(panel_T)

    print(f"\n=== {label}  (angles {angle_range[0]}-{angle_range[1]} deg, seed {seed}) ===")
    print(f"{'panel':6s} {'primary':>7s} {'areaA':>8s} {'areaB':>8s} {'shared':>7s} "
          f"{'dA':>8s} {'dB':>8s}  {'serves':>10s}")
    rows = []
    n_both = 0
    for gi, p in enumerate(panels):
        aA = wall_coverage_area(table, p, "A")
        aB = wall_coverage_area(table, p, "B")
        shared = min(aA, aB) / max(aA, aB, 1e-12)

        # --- ABLATION: clear this one panel, re-render, difference both walls ----------
        q = panel_T.copy()
        q[gi] = 1.0
        abl = renderer.render_color_np(q)
        dA = float(np.abs(abl["A"] - base["A"]).mean())
        dB = float(np.abs(abl["B"] - base["B"]).mean())

        serves = []
        if dA > ABLATE_KNEE:
            serves.append("A")
        if dB > ABLATE_KNEE:
            serves.append("B")
        both = len(serves) == 2
        n_both += both
        print(f"{p.name:6s} {primary_wall_of(layout, table, p):>7s} {aA:8.4f} {aB:8.4f} "
              f"{shared:7.2f} {dA:8.5f} {dB:8.5f}  {'+'.join(serves) or '-':>10s}"
              f"{'  <-- BOTH' if both else ''}")
        rows.append(dict(panel=p.name, area_A=aA, area_B=aB, shared_ratio=shared,
                         ablate_dA=dA, ablate_dB=dB, serves_both=both,
                         primary=primary_wall_of(layout, table, p)))

    prim = {p.name: primary_wall_of(layout, table, p) for p in panels}
    good, bad = colour_agreeing_duty(renderer, panel_T, panels, targets,
                                     layout.white_threshold, prim=prim,
                                     match_tol=FR.MATCH_TOL_REPORT, match_metric="lab")
    joint = joint_intersection_pct(fragments, table, panels)
    gm, bm = 0.5 * (good["A"] + good["B"]), 0.5 * (bad["A"] + bad["B"])

    print(f"\n  panels serving BOTH walls (ablation): {n_both}/{len(panels)}"
          f"   {'OK' if n_both >= 2 else '*** FEWER THAN 2 ***'}")
    print(f"  mean shared ratio                   : "
          f"{np.mean([r['shared_ratio'] for r in rows]):.2f}")
    print(f"  colour-agreeing duty  good          : A={good['A']:5.1f}%  B={good['B']:5.1f}%"
          f"   mean={gm:5.2f}%")
    print(f"  wrong-tone bleed      bad           : A={bad['A']:5.1f}%  B={bad['B']:5.1f}%"
          f"   mean={bm:5.2f}%")
    print(f"  good/bad ratio                      : {gm / bm if bm else float('inf'):.2f}"
          f"    (compat ~0.9, arbitrary ~0.3)")
    print(f"  joint@0.2 (COLOUR-BLIND, context)   : {joint[0.2]:.1f}%")

    return dict(label=label, seed=seed, angle_range=list(angle_range),
                n_panels=len(panels), n_serving_both=n_both,
                mean_shared_ratio=float(np.mean([r["shared_ratio"] for r in rows])),
                good_A=good["A"], good_B=good["B"], bad_A=bad["A"], bad_B=bad["B"],
                good_mean=gm, bad_mean=bm,
                good_bad_ratio=(gm / bm if bm else None),
                joint_02=joint[0.2], panels=rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cands = FR.write_targets()
    out = [diagnose(label, ka, kb, cands, seed=2) for label, ka, kb in FR.PAIRS]
    (OUT / "duty_diagnostic.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}/duty_diagnostic.json")


if __name__ == "__main__":
    main()
