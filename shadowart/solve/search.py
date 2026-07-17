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

from ..targets import color as C
from .. import metrics as _metrics
from . import decompose
from . import panel_search


# Higher score = better. `joint_thresh` selects which threshold of
# `joint_intersection_pct` to penalise (0.3 = the strictest "genuine double duty" band,
# so cross-talk that is merely incidental overlap is not over-penalised).
DEFAULT_WEIGHTS = {"ssim": 1.0, "edge": 0.5, "crosstalk": 0.25, "joint_thresh": 0.3}


def score_layout(accuracy, joint_pct, weights=None):
    """Composite scalar (higher is better) ranking a whole layout across restarts.

    `accuracy`  : `metrics.evaluate_wall_accuracy` output -- {'A': {...ssim,
                  edge_fidelity...}, 'B': {...}}.
    `joint_pct` : `panel_search.joint_intersection_pct` output -- {0.1: pct, 0.2: pct,
                  0.3: pct} -- or None (then the cross-talk term is dropped).
    `weights`   : override any of DEFAULT_WEIGHTS.

        score = w_ssim*(ssimA+ssimB) + w_edge*(edgeA+edgeB) - w_crosstalk*(joint_pct[thr]/100)

    Reused by both greedies' restart loops so the two searches optimise the same quantity.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    a, b = accuracy["A"], accuracy["B"]
    s = w["ssim"] * (a["ssim"] + b["ssim"])
    s += w["edge"] * (a["edge_fidelity"] + b["edge_fidelity"])
    if joint_pct:
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
    return {
        "stack_colorid": stack_colorid, "opacity": opacity, "fragments": fragments,
        "resolved": resolved, "stack_depths": stack_depths, "budget_stats": budget_stats,
        "stack_intensity": stack_intensity, "pred_rgb": pred_rgb,
        "accuracy": accuracy, "joint_pct": joint_pct,
        "score": score_layout(accuracy, joint_pct, weights), "seed": seed,
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
            scene, panel_count, mode="deliberate", K=greedy_K, targets=targets, seed=pseed)
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
