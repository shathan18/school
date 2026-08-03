"""Multi-run search + best-selection over the two greedy solvers.

Both greedies in this project are single-pass and seed-driven:
  * shard placement  -- `decompose.fragment_shards_overlap` picks each shard's host
    depth-plane once per run (and its damage-minimising branch is only active when
    `damage_weight > 0`);
  * panel layout     -- `panel_search.build_panels_greedy` constructs a panel set once
    per seed.

Neither had a "restart N times and keep the best" mechanism, and there was no single
scalar to rank whole layouts by. This module adds exactly that, WITHOUT touching either
greedy: it drives them, scores each result with `score_layout`, and returns the winning
run's full export bundle (the same tuple `_run_color_overlap` already consumes) so the
rest of the CLI is unchanged.

`score_layout` deliberately drives on SSIM + edge-fidelity (structural detail) and
penalises the high-threshold joint-intersection (unwanted cross-wall bleed in overlap
mode); RMSE/PSNR are excluded because -- per `metrics.py`'s own docstring -- they reward
blur.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from ..targets import color as C
from .. import metrics as _metrics
from . import decompose
from . import panel_search


# Higher score = better. `joint_thresh` selects which threshold of
# `joint_intersection_pct` to penalise (0.3 = the strictest "genuine double duty" band,
# so cross-talk that is merely incidental overlap is not over-penalised).
# `duty`/`bleed` drive the COLOUR-AWARE path (see score_layout).
DEFAULT_WEIGHTS = {"ssim": 1.0, "edge": 0.5, "crosstalk": 0.25, "joint_thresh": 0.3,
                   "duty": 0.5, "bleed": 0.5}


def colour_agreeing_duty(renderer, panel_T, panels, targets, white_thr, prim=None,
                         match_tol=0.30, dark_thr=0.05, match_metric="rgb"):
    """Split cross-wall bleed into the GOOD and BAD halves, per wall, as % of that wall's subject.

    Renders only the panels whose primary wall is the OTHER one -- i.e. pure cross-talk -- then
    of the subject pixels it darkens:
      good = it arrived in a colour that wall actually wants (||pred - target|| < match_tol)
      bad  = it arrived in the wrong colour (visible contamination)

    This is the corrected measure of `corrections_note.md` 3: the geometric
    `panel_search.joint_intersection_pct` counts "landed on content" regardless of colour, which
    is exactly the number that read 27-31% and collapsed to ~3% once colour was required."""
    from ..geometry.projection import primary_wall_of          # local: avoids a circular import
    if prim is None:
        prim = {}
    good, bad = {}, {}
    for wall in renderer.scene.walls:
        q = panel_T.copy()
        for gi, p in enumerate(panels):
            if prim.get(p.name) == wall:
                q[gi] = 1.0                                     # drop the panels that BUILD it
        xr = renderer.render_color_np(q)[wall]
        subj = C.subject_mask(targets[wall], white_thr)
        denom = max(int(subj.sum()), 1)
        onsub = ((1.0 - xr.mean(-1)) > dark_thr) & subj
        if onsub.any():
            if match_metric == "lab":                          # perceptual dE, matches the optimiser gate
                d = C.delta_e(xr[onsub], targets[wall][onsub])
            else:
                d = np.sqrt(((xr[onsub] - targets[wall][onsub]) ** 2).sum(1))
            good[wall] = 100.0 * float((d < match_tol).sum()) / denom
            bad[wall] = 100.0 * float((d >= match_tol).sum()) / denom
        else:
            good[wall] = bad[wall] = 0.0
    return good, bad


def score_layout(accuracy, joint_pct, weights=None, duty=None, bleed=None):
    """Composite scalar (higher is better) ranking a whole layout across restarts.

    `accuracy`  : `metrics.evaluate_wall_accuracy` output -- {'A': {...ssim,
                  edge_fidelity...}, 'B': {...}}.
    `joint_pct` : `panel_search.joint_intersection_pct` output -- {0.1: pct, 0.2: pct,
                  0.3: pct} -- or None.
    `duty`/`bleed`: `colour_agreeing_duty` output ({'A': pct, 'B': pct} each) or None.
    `weights`   : override any of DEFAULT_WEIGHTS.

    COLOUR-AWARE path (when `duty` is given) -- preferred:

        score = w_ssim*(ssimA+ssimB) + w_edge*(edgeA+edgeB)
                + w_duty*mean(duty)/100 - w_bleed*mean(bleed)/100

    Genuine, colour-agreeing double duty is a BONUS; only wrong-colour bleed is penalised. The
    legacy path below instead subtracted the colour-BLIND `joint_intersection_pct`, which
    penalised real double duty -- i.e. it fought the per-shard signed credit in
    `decompose._shard_damage`, which rewards exactly that. Both levels now optimise the same
    quantity. Units are consistent here: duty and bleed are both % of the wall's subject.

    LEGACY path (duty=None) reproduces the previous behaviour exactly, so callers that cannot
    afford the extra cross-talk-only render (e.g. panel_search) are unchanged.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    a, b = accuracy["A"], accuracy["B"]
    s = w["ssim"] * (a["ssim"] + b["ssim"])
    s += w["edge"] * (a["edge_fidelity"] + b["edge_fidelity"])
    if duty is not None:
        s += w["duty"] * (duty.get("A", 0.0) + duty.get("B", 0.0)) / 2.0 / 100.0
        if bleed is not None:
            s -= w["bleed"] * (bleed.get("A", 0.0) + bleed.get("B", 0.0)) / 2.0 / 100.0
    elif joint_pct:
        s -= w["crosstalk"] * (joint_pct.get(w["joint_thresh"], 0.0) / 100.0)
    return float(s)


def _restart_iter(restarts, time_budget, base_seed):
    """Yield (i, base_seed+i) until `restarts` reached OR `time_budget` seconds elapse,
    whichever comes first; always yields at least once. `time.perf_counter` is correct in
    shipped application code (the no-wallclock rule is a workflow-script constraint only)."""
    t0 = time.perf_counter()
    i = 0
    while True:
        yield i, base_seed + i
        i += 1
        if restarts and i >= restarts:
            return
        if time_budget and (time.perf_counter() - t0) >= time_budget:
            return
        if not restarts and not time_budget:      # no stop condition given -> single run
            return


def _run_once_shards(scene, table, targets, names, renderer, seed,
                     damage_weight, credit_weight, match_tol, weights):
    """One shard-placement pass -> a scored export bundle. Mirrors exactly what
    `_run_color_overlap` and `panel_search.run_layout` already do per run."""
    (stack_colorid, opacity, fragments, resolved, stack_depths,
     budget_stats, stack_intensity) = decompose.fragment_shards_overlap(
        scene, table, targets, names=names, white_thr=scene.white_threshold,
        max_stack=scene.color_max_stack, seed=seed,
        damage_weight=damage_weight, credit_weight=credit_weight, match_tol=match_tol,
        diagonal_frac=scene.solve.diagonal_frac)
    panel_T = C.stack_transmit_lut(names, stack_colorid, stack_intensity)
    pred_rgb = renderer.render_color_np(panel_T)
    accuracy = _metrics.evaluate_wall_accuracy(targets, pred_rgb)
    joint_pct = panel_search.joint_intersection_pct(fragments, table, scene.panels)
    # colour-aware cross-talk terms: reward the bleed that arrives in a wanted colour, penalise
    # only the rest. Costs one extra cross-talk-only render per run (see colour_agreeing_duty).
    from ..geometry.projection import primary_wall_of
    prim = {p.name: primary_wall_of(scene, table, p) for p in scene.panels}
    duty, bleed = colour_agreeing_duty(renderer, panel_T, scene.panels, targets,
                                       scene.white_threshold, prim=prim, match_tol=match_tol)
    return {
        "stack_colorid": stack_colorid, "opacity": opacity, "fragments": fragments,
        "resolved": resolved, "stack_depths": stack_depths, "budget_stats": budget_stats,
        "stack_intensity": stack_intensity, "pred_rgb": pred_rgb,
        "accuracy": accuracy, "joint_pct": joint_pct, "duty": duty, "bleed": bleed,
        "score": score_layout(accuracy, joint_pct, weights, duty=duty, bleed=bleed),
        "seed": seed,
    }


def multi_run_shards(scene, table, targets, names, renderer, *,
                     restarts=1, time_budget=None, base_seed=None,
                     damage_weight=0.0, credit_weight=None, match_tol=0.30,
                     weights=None):
    """Restart shard placement (`decompose.fragment_shards_overlap`) up to `restarts` times
    (or until `time_budget` seconds), keep the best by `score_layout`. `table`/`renderer`
    are fixed for all runs. Returns the winning run's bundle plus `n_runs` (the export
    bundle keys match what `_run_color_overlap` destructures)."""
    base_seed = scene.solve.seed if base_seed is None else base_seed
    best, n_runs = None, 0
    for _, seed in _restart_iter(restarts, time_budget, base_seed):
        n_runs += 1
        run = _run_once_shards(scene, table, targets, names, renderer, seed,
                               damage_weight, credit_weight, match_tol, weights)
        if best is None or run["score"] > best["score"]:
            best = run
    best["n_runs"] = n_runs
    return best


def multi_run_panels(scene, targets, names, *,
                     panel_restarts=1, shard_restarts=1,
                     panel_time_budget=None, shard_time_budget=None,
                     base_seed=None, panel_count=None,
                     damage_weight=0.0, credit_weight=None, match_tol=0.30,
                     weights=None, greedy_K=12):
    """Outer loop over panel layouts; inner loop over shard placements.

    For each of `panel_restarts` layouts from `panel_search.build_panels_greedy`, rebuild
    the projection table + renderer for that layout (same `dataclasses.replace(scene,
    panels=...)` pattern as `panel_search.run_layout`), then nest `multi_run_shards` and
    keep the globally best-scoring (layout, shard-run) pair. `panel_count` defaults to the
    current scene's panel count. Returns
        {panels, table, renderer, shard_best, score, panel_seed, n_layouts}.
    """
    from ..geometry.projection import build_projection_table
    from ..forward.renderer import Renderer

    base_seed = scene.solve.seed if base_seed is None else base_seed
    panel_count = len(scene.panels) if panel_count is None else panel_count
    best, n_layouts = None, 0
    for _, pseed in _restart_iter(panel_restarts, panel_time_budget, base_seed):
        n_layouts += 1
        panels, _scores = panel_search.build_panels_greedy(
            scene, panel_count, mode="deliberate", K=greedy_K, targets=targets, seed=pseed,
            anchor_range=scene.solve.search_anchor_range,
            standoff=scene.solve.search_standoff,
            mag_cap=scene.solve.search_mag_cap,
            u_size_range=scene.solve.search_u_size_range,
            v_range=scene.solve.search_v_range)
        if not panels:                            # degenerate seed produced no valid layout
            continue
        layout_scene = dataclasses.replace(scene, panels=panels)
        table = build_projection_table(layout_scene)
        renderer = Renderer(layout_scene, table)
        shard_best = multi_run_shards(
            layout_scene, table, targets, names, renderer,
            restarts=shard_restarts, time_budget=shard_time_budget, base_seed=base_seed,
            damage_weight=damage_weight, credit_weight=credit_weight,
            match_tol=match_tol, weights=weights)
        if best is None or shard_best["score"] > best["score"]:
            best = {"panels": panels, "table": table, "renderer": renderer,
                    "shard_best": shard_best, "score": shard_best["score"],
                    "panel_seed": pseed}
    if best is None:
        raise RuntimeError("panel search produced no valid layout on any seed")
    best["n_layouts"] = n_layouts
    return best
