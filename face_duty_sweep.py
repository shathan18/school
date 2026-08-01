"""Sweep the levers that control GENUINE double duty, then verify the winner physically.

WHY THIS EXISTS. The first face render (`face_render300.py`) optimised wall fidelity and
never measured overlap. `face_duty_check.py` then showed the "winning" Poe/Dostoevsky
layout was a poor sculpture: only 2/14 panels affected both walls, FIVE panels affected
neither (dead perspex), and colour-agreeing duty was good 0.27% / bad 2.73%, a good/bad
ratio of 0.10 -- worse than the ~0.3 an ARBITRARY image pair gets. Fidelity was bought by
letting each panel serve one wall. That is two independent shadow puppets sharing a room,
not an intersection piece.

DIAGNOSIS. The two-tone sources need few shards (204 total, because white = clear = no
shard). Spread over 14 panels that is ~15 shards each, so most panels end up empty or
single-purpose. Fewer panels, each more square-on to both walls, should force the same
shards to share.

LEVERS SWEPT (everything else held fixed):
  panel_count   14 -> 6.  Fewer panels means each carries more shards and cannot afford to
                serve only one wall.
  angle_range   (30,60) vs (40,50) vs (43,47). A panel at 45 deg is equally face-on to both
                walls, so its projected footprints match and one shard can register on both
                images. 30 deg and 60 deg panels are strongly biased to one wall.
  credit_weight 1.0 -> 2.0 -> 3.0. `decompose._shard_damage`'s signed credit for a shard
                that lands in a tone the OTHER wall also wants.

NOT swept, because shadowart-noise.md already records them as verified dead ends:
`spill_weight` in build_panels_greedy (raises good AND bad together, costs SSIM),
`outline_masks`/`outline_protect_weight` (null), damage-only "harm_only" (collapses panels).

METRICS. Ranked on duty, but fidelity is reported alongside so a layout cannot win by
destroying the images:
  n_both      panels that change BOTH walls under single-panel ABLATION -- the physical
              test, not a geometric guess. HARD REQUIREMENT: >= 2.
  n_dead      panels that change NEITHER wall. Should be 0; these are wasted material.
  good/bad    `colour_agreeing_duty` at the fixed dE 25 reporting gate.
  face_det    face-box detail retention (does the face still read?).
`joint_intersection_pct` is deliberately absent from the ranking: it is colour-blind and
reads 100% for every layout (shadowart-noise.md).
"""
from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np

from shadowart import metrics as _metrics
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

# Ablation: a panel "contributes" to a wall if clearing it moves that wall's mean RGB by
# more than this. 0.002 ~= 0.5/255 -- above float noise, below visibility.
ABLATE_KNEE = 0.002
MIN_PANELS_BOTH = 2                       # the hard requirement


def build(pair, seed, panel_count, angle_range, credit_weight):
    """Build one full layout + shard decomposition. Returns everything needed to measure."""
    label, ka, kb = pair
    scene = load_scene(FR.SCENE)
    scene = dataclasses.replace(scene, color_palette=C.PALETTES["noir"])
    names = C.palette_names(scene.color_palette)
    wr = scene.solve.wall_res
    targets = {
        "A": C.load_color_target(str(FR.TGT / f"{ka}.png"), wr, white_thr=scene.white_threshold),
        "B": C.load_color_target(str(FR.TGT / f"{kb}.png"), wr, white_thr=scene.white_threshold),
    }
    panels, _ = build_panels_greedy(scene, count=panel_count, mode="deliberate",
                                    K=FR.K_CANDIDATES, targets=targets, seed=seed,
                                    angle_deg_range=angle_range)
    layout = dataclasses.replace(scene, panels=panels)
    table = build_projection_table(layout)
    renderer = Renderer(layout, table)
    sc, opacity, fragments, resolved, sd, bs, si = decompose.fragment_shards_overlap(
        layout, table, targets, names=names, white_thr=layout.white_threshold,
        max_stack=layout.color_max_stack, seed=seed,
        shard_budget=FR.SHARD_BUDGET_PER_WALL,
        damage_weight=FR.DAMAGE_WEIGHT, credit_weight=credit_weight,
        match_metric=FR.MATCH_METRIC, match_tol=FR.MATCH_TOL)
    panel_T = C.stack_transmit_lut(names, sc, si)
    return dict(layout=layout, table=table, renderer=renderer, panels=panels,
                targets=targets, panel_T=panel_T, names=names, fragments=fragments,
                opacity=opacity, stack_colorid=sc, budget=bs)


def ablate(b, verbose=False):
    """Single-panel ablation -- the physical proof of which walls each panel really serves.

    Re-render with exactly one panel cleared and difference BOTH walls. This runs the actual
    forward renderer, so it cannot be satisfied by bookkeeping: a panel only registers as
    serving a wall if removing it visibly changes that wall."""
    base = b["renderer"].render_color_np(b["panel_T"])
    rows = []
    for gi, p in enumerate(b["panels"]):
        q = b["panel_T"].copy()
        q[gi] = 1.0
        abl = b["renderer"].render_color_np(q)
        dA = float(np.abs(abl["A"] - base["A"]).mean())
        dB = float(np.abs(abl["B"] - base["B"]).mean())
        aA = wall_coverage_area(b["table"], p, "A")
        aB = wall_coverage_area(b["table"], p, "B")
        rows.append(dict(panel=p.name, ablate_dA=dA, ablate_dB=dB,
                         serves_A=dA > ABLATE_KNEE, serves_B=dB > ABLATE_KNEE,
                         shared_ratio=min(aA, aB) / max(aA, aB, 1e-12),
                         primary=primary_wall_of(b["layout"], b["table"], p)))
        if verbose:
            s = ("A" if rows[-1]["serves_A"] else "") + ("B" if rows[-1]["serves_B"] else "")
            print(f"    {p.name:5s} primary={rows[-1]['primary']}  dA={dA:.5f} dB={dB:.5f}"
                  f"  shared={rows[-1]['shared_ratio']:.2f}  serves={s or '-':2s}"
                  f"{'   <-- BOTH' if s == 'AB' else ''}")
    return base, rows


def measure(b, base, rows):
    prim = {p.name: primary_wall_of(b["layout"], b["table"], p) for p in b["panels"]}
    good, bad = colour_agreeing_duty(
        b["renderer"], b["panel_T"], b["panels"], b["targets"],
        b["layout"].white_threshold, prim=prim,
        match_tol=FR.MATCH_TOL_REPORT, match_metric="lab")
    acc = _metrics.evaluate_wall_accuracy(b["targets"], base)
    fd = {w: FR.detail_retention_wall(b["targets"][w], base[w],
                                      FR.face_mask_wall(b["targets"][w], f))
          for w, f in (("A", b["_faceA"]), ("B", b["_faceB"]))}
    gm, bm = 0.5 * (good["A"] + good["B"]), 0.5 * (bad["A"] + bad["B"])
    n_a = int(b["budget"].get("A", {}).get("achieved", 0))
    n_b = int(b["budget"].get("B", {}).get("achieved", 0))
    return dict(
        n_both=sum(r["serves_A"] and r["serves_B"] for r in rows),
        n_dead=sum((not r["serves_A"]) and (not r["serves_B"]) for r in rows),
        n_panels=len(rows),
        mean_shared=float(np.mean([r["shared_ratio"] for r in rows])),
        good_mean=gm, bad_mean=bm,
        good_bad=(gm / bm if bm else None),
        ssim=0.5 * (acc["A"]["ssim"] + acc["B"]["ssim"]),
        edge=0.5 * (acc["A"]["edge_fidelity"] + acc["B"]["edge_fidelity"]),
        face_det=0.5 * (fd["A"] + fd["B"]),
        shards=n_a + n_b,
        joint_02=joint_intersection_pct(b["fragments"], b["table"], b["panels"])[0.2],
        panels=rows,
    )


def run(pair, seed, panel_count, angle_range, credit_weight, cands, verbose=False):
    b = build(pair, seed, panel_count, angle_range, credit_weight)
    b["_faceA"], b["_faceB"] = cands[pair[1]].face, cands[pair[2]].face
    base, rows = ablate(b, verbose=verbose)
    m = measure(b, base, rows)
    m.update(label=pair[0], seed=seed, panel_count=panel_count,
             angle_range=list(angle_range), credit_weight=credit_weight)
    return m, b, base


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cands = FR.write_targets()
    pair = FR.PAIRS[0]                      # Poe + Dostoevsky: the layout that must be fixed

    print("=" * 100)
    print("DUTY SWEEP -- forcing genuine intersection on the two-tone face pair")
    print(f"  hard requirement: >= {MIN_PANELS_BOTH} panels changing BOTH walls under ablation")
    print("=" * 100)
    print(f"{'panels':>6s} {'angles':>9s} {'cw':>4s} | {'n_both':>6s} {'dead':>4s} "
          f"{'shared':>6s} | {'good%':>6s} {'bad%':>6s} {'g/b':>5s} | "
          f"{'SSIM':>5s} {'face':>5s} {'shards':>6s}")
    print("-" * 100)

    rows = []
    grid = itertools.product([14, 10, 8, 6], [(30, 60), (40, 50), (43, 47)], [1.0, 3.0])
    for pc, ar, cw in grid:
        m, _b, _base = run(pair, 2, pc, ar, cw, cands)
        rows.append(m)
        ok = "OK " if m["n_both"] >= MIN_PANELS_BOTH else "*** "
        print(f"{pc:>6d} {str(ar):>9s} {cw:>4.1f} | {m['n_both']:>6d} {m['n_dead']:>4d} "
              f"{m['mean_shared']:>6.2f} | {m['good_mean']:>6.2f} {m['bad_mean']:>6.2f} "
              f"{(m['good_bad'] or 0):>5.2f} | {m['ssim']:>5.3f} {m['face_det']:>5.3f} "
              f"{m['shards']:>6d} {ok}")

    (OUT / "duty_sweep.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "panels"} for r in rows], indent=2),
        encoding="utf-8")

    # Rank: must clear the hard requirement and keep the face readable, then maximise duty.
    ok = [r for r in rows if r["n_both"] >= MIN_PANELS_BOTH and r["face_det"] >= 0.60]
    ok.sort(key=lambda r: -(r["good_bad"] or 0))
    print("\nTop configs clearing n_both>=2 AND face_det>=0.60, ranked by good/bad:")
    for r in ok[:5]:
        print(f"  panels={r['panel_count']:>2d} angles={tuple(r['angle_range'])} "
              f"cw={r['credit_weight']:.1f}  ->  n_both={r['n_both']} dead={r['n_dead']} "
              f"good={r['good_mean']:.2f}% bad={r['bad_mean']:.2f}% g/b={r['good_bad']:.2f} "
              f"SSIM={r['ssim']:.3f} face={r['face_det']:.3f}")
    if not ok:
        print("  NONE. No configuration produced a real intersection piece.")
    print(f"\nwrote {OUT}/duty_sweep.json")


if __name__ == "__main__":
    main()
