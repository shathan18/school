"""Full 300-shard render + colour-agreeing compatibility numbers for one pair.

Given (target_a, target_b, label): builds panels with the same greedy-diagonal recipe
as sweep_panels.py, runs fragment_shards_overlap at ~300-shard budget, and reports:

  SSIM_A/B, edge_A/B     (per-wall fidelity)
  good_A/B (%)           colour-agreeing double duty  (colour_agreeing_duty)
  bad_A/B (%)            wrong-colour cross-talk bleed
  ratio                  mean(good) / mean(bad)  -- palette compat surrogate
                         (report_team.md: compat pairs ~0.9, arbitrary ~0.3)
  joint@0.2 (%)          colour-blind geometric intersection (context only)

Also writes preview_final.png and scene_interactive.html per pair, and dumps
metrics.json into out_pair_selection/pairs/<label>/.
"""
from __future__ import annotations

import dataclasses
import json
import time
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
OUT_ROOT = Path("out_pair_selection/pairs")

# Same host-selection recipe as sweep_panels.py + render_best_panels.py.
ANGLE_RANGE = (30, 60)
K_CANDIDATES = 16
DAMAGE_WEIGHT = 0.5
CREDIT_WEIGHT = 1.0
PANEL_COUNT = 14              # sweep_panels.py peak for compatible pairs was pc>=12
SEED = 2
MATCH_METRIC = "lab"
MATCH_TOL_SOLVER = 20.0       # palette-compat pairs: loose enough per out_noise_study
MATCH_TOL_REPORT = 25.0       # fixed reporting gate per shadowart-noise memory


def render_pair(a_path: str, b_path: str, label: str) -> dict:
    out = OUT_ROOT / label
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"\n=== PAIR: {label} ===")
    print(f"  A = {a_path}")
    print(f"  B = {b_path}")
    scene = load_scene(SCENE)
    wr = scene.solve.wall_res
    names = ["clear"] + C.CMYK

    targets = {
        "A": C.load_color_target(a_path, wr, white_thr=scene.white_threshold),
        "B": C.load_color_target(b_path, wr, white_thr=scene.white_threshold),
    }
    panels, _ = build_panels_greedy(
        scene, count=PANEL_COUNT, mode="deliberate", K=K_CANDIDATES,
        targets=targets, seed=SEED, angle_deg_range=ANGLE_RANGE,
    )
    scene_layout = dataclasses.replace(scene, panels=panels)
    table = build_projection_table(scene_layout)
    renderer = Renderer(scene_layout, table)

    sc, opacity, fragments, resolved, sd, bs, si = decompose.fragment_shards_overlap(
        scene_layout, table, targets, names=names,
        white_thr=scene_layout.white_threshold,
        max_stack=scene_layout.color_max_stack, seed=SEED,
        damage_weight=DAMAGE_WEIGHT, credit_weight=CREDIT_WEIGHT,
        match_metric=MATCH_METRIC, match_tol=MATCH_TOL_SOLVER,
    )
    panel_T = C.stack_transmit_lut(names, sc, si)
    pred_rgb = renderer.render_color_np(panel_T)
    acc = _metrics.evaluate_wall_accuracy(targets, pred_rgb)
    joint = joint_intersection_pct(fragments, table, panels)

    prim = {p.name: primary_wall_of(scene_layout, table, p) for p in panels}
    good, bad = colour_agreeing_duty(
        renderer, panel_T, panels, targets, scene_layout.white_threshold,
        prim=prim, match_tol=MATCH_TOL_REPORT, match_metric="lab",
    )
    good_mean = 0.5 * (good["A"] + good["B"])
    bad_mean = 0.5 * (bad["A"] + bad["B"])
    ratio = good_mean / bad_mean if bad_mean > 0 else float("inf")

    n_shards = int(bs.get("A", {}).get("achieved", 0)) + int(bs.get("B", {}).get("achieved", 0))
    elapsed = time.perf_counter() - t0

    metrics = dict(
        label=label,
        target_a=a_path, target_b=b_path,
        n_shards=n_shards, elapsed_s=round(elapsed, 1),
        ssim_A=float(acc["A"]["ssim"]), ssim_B=float(acc["B"]["ssim"]),
        edge_A=float(acc["A"]["edge_fidelity"]), edge_B=float(acc["B"]["edge_fidelity"]),
        good_A=float(good["A"]), good_B=float(good["B"]),
        bad_A=float(bad["A"]), bad_B=float(bad["B"]),
        good_mean=float(good_mean), bad_mean=float(bad_mean),
        good_bad_ratio=float(ratio) if ratio != float("inf") else None,
        joint_01=float(joint[0.1]), joint_02=float(joint[0.2]), joint_03=float(joint[0.3]),
    )

    print(f"  shards          : {n_shards}")
    print(f"  SSIM            : A={acc['A']['ssim']:.3f}  B={acc['B']['ssim']:.3f}")
    print(f"  edge fidelity   : A={acc['A']['edge_fidelity']:.3f}  B={acc['B']['edge_fidelity']:.3f}")
    print(f"  good  duty (%)  : A={good['A']:5.1f}  B={good['B']:5.1f}  mean={good_mean:5.2f}")
    print(f"  bad  bleed (%)  : A={bad['A']:5.1f}  B={bad['B']:5.1f}  mean={bad_mean:5.2f}")
    print(f"  good/bad ratio  : {ratio:.2f}   (compat pairs ~=0.9, arbitrary ~=0.3)")
    print(f"  joint@0.2       : {joint[0.2]:.1f}%  (colour-BLIND, context only)")
    print(f"  elapsed         : {elapsed:.1f}s")

    save_color_comparison(targets, pred_rgb, out / "preview_final.png")
    np.save(out / "opacity.npy", opacity)
    np.save(out / "stack_colorid.npy", sc)

    stack_pieces = decompose.panel_stack_pieces(scene_layout, sc, names)
    poly_channel = {id(poly): ch for items in stack_pieces.values() for poly, ch, _s in items}
    flat_pieces = {name: [poly for poly, _ch, _s in items]
                   for name, items in stack_pieces.items()}
    stack_color_of = lambda panel, poly: tuple(C.display_rgb(poly_channel.get(id(poly), "clear")))
    build_interactive(scene_layout, table, opacity, None,
                      out / "scene_interactive.html", rays=40, auto_open=False,
                      wall_rgb=pred_rgb, pieces=flat_pieces, color_of=stack_color_of)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  wrote {out}/preview_final.png, scene_interactive.html, metrics.json")
    return metrics


# ------------------------------------------------------------------- DRIVER
def load_top_pairs(k: int) -> list[tuple[str, str, str]]:
    """Read out_pair_selection/pair_scores.tsv and pull the top-k rows."""
    tsv = Path("out_pair_selection/pair_scores.tsv")
    lines = tsv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    ia, ib = header.index("a"), header.index("b")
    ipa, ipb = header.index("path_a"), header.index("path_b")
    out = []
    for row in lines[1:1 + k]:
        f = row.split("\t")
        a_name, b_name = f[ia], f[ib]
        # short label: combine the two truncated stems
        def _short(x): return x.replace("katsushika_hokusai_", "").split("_from_")[0][:26]
        label = f"{_short(a_name)}__{_short(b_name)}"
        out.append((f[ipa], f[ipb], label))
    return out


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pairs = load_top_pairs(3)
    print(f"Rendering top-{len(pairs)} pairs at PC={PANEL_COUNT}, seed={SEED}, dw={DAMAGE_WEIGHT}, "
          f"cw={CREDIT_WEIGHT}, match_metric={MATCH_METRIC}, match_tol_solver={MATCH_TOL_SOLVER}, "
          f"match_tol_report={MATCH_TOL_REPORT}")
    results = []
    for pa, pb, label in pairs:
        m = render_pair(pa, pb, label)
        results.append(m)

    # Ranked table + save
    results.sort(key=lambda r: -(r["good_mean"] - r["bad_mean"]))
    summary = Path("out_pair_selection/pair_render_summary.md")
    lines = [
        "# Pair render results (300-shard, lab-gated)",
        "",
        f"Recipe: PC={PANEL_COUNT}, seed={SEED}, dw={DAMAGE_WEIGHT}, cw={CREDIT_WEIGHT}, "
        f"match_metric='{MATCH_METRIC}', solver_tol={MATCH_TOL_SOLVER}, report_tol={MATCH_TOL_REPORT}.",
        "",
        "Ranked by `good_mean - bad_mean` (net colour-agreeing double duty).",
        "",
        "| # | label | SSIM A/B | edge A/B | good A/B | bad A/B | good/bad | shards | elapsed |",
        "|---|-------|:--------:|:--------:|:--------:|:-------:|:--------:|:------:|:-------:|",
    ]
    for i, r in enumerate(results, 1):
        gb = f"{r['good_bad_ratio']:.2f}" if r["good_bad_ratio"] is not None else "inf"
        lines.append(
            f"| {i} | {r['label']} | "
            f"{r['ssim_A']:.3f}/{r['ssim_B']:.3f} | "
            f"{r['edge_A']:.3f}/{r['edge_B']:.3f} | "
            f"{r['good_A']:.1f}/{r['good_B']:.1f}% | "
            f"{r['bad_A']:.1f}/{r['bad_B']:.1f}% | "
            f"{gb} | {r['n_shards']} | {r['elapsed_s']}s |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- **good/bad ratio** is the palette-compat surrogate. Reference (from `report_team.md` /",
        "  `out_noise_study/`): palette-compatible pairs land ~0.88..0.93, arbitrary pairs ~0.27..0.35.",
        "- **SSIM/edge** are per-wall fidelity. Absolute SSIM ~=0.68..0.72 is normal at 300 shards.",
        "- **joint@0.2** is colour-BLIND (reads ~100% for every layout) - context only, not a ranker.",
    ]
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
