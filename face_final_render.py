"""FINAL RENDER -- a piece that actually intersects, with the proof attached.

This supersedes the render in `face_render300.py`, whose output was not an intersection
sculpture. See `_why_this_replaces_face_render300` below.

WHAT IS ENFORCED HERE
  * GRAY_L paper (`floor=0.75`). Light tone becomes material instead of bare wall, so the
    whole piece can participate. This is the single change that turns 2/10 panels serving
    both walls into 8-9/10.
  * A HARD ASSERT: at least MIN_BOTH panels must change BOTH walls under single-panel
    ablation, or the run raises. Not a warning, not a printed number -- the render fails.
  * The ablation table is written into metrics.json for every run, so the claim "these
    planes serve both images" is always backed by a re-runnable measurement.

WHY THE ABLATION IS THE PROOF, AND THE PROJECTION IS REAL
  `Renderer.render_color_np` loops every panel for every wall, warping each through its own
  projective homography (`H_wp`, from `homography_panel_to_wall` for THAT wall's light),
  blurring by that panel's physical penumbra sigma, and multiplying transmittances. There is
  no family shortcut and no per-wall panel subset anywhere in it -- `projection.py`'s module
  docstring says the same ("There's no per-panel label saying which is which"), and
  `primary_wall_of` is derived from projected area after the fact, never declared.
  The ablation test does not take any of that on trust: it clears ONE panel, re-runs the
  real forward renderer, and differences both walls. A panel is only credited with serving a
  wall if removing it visibly changes that wall.

THE TRADE, STATED PLAINLY (measured, `out_faces_hc/duty/density_scan.json`)
    config                              shards  panels serving both  good%   g/b   face_det
    clear paper (the old render)          204          2/10           0.58   0.72    0.743
    GRAY_L paper (this one)               343          8/10          16.55   1.66    0.378
    GRAY_L paper, 6x the shards          2353          9/10          17.61   1.85    0.505
  The face is NOT legible at 300 shards once the piece really intersects, and more shards do
  not rescue it: face_det saturates near 0.55 by ~7800 shards. Conversely the clear-paper
  face_det is already saturated at 204 shards (0.743 -> 0.746 at 4377), so its legibility was
  never a resolution win -- it was the absence of half the sculpture.

GRAYSCALE, CORRECTED. I previously flagged that grayscale forfeits the colour-compatibility
result. For double duty that was backwards: under `noir` both walls want the same four
neutral greys, and this pair reaches good/bad 1.66-2.08 -- roughly double the best COLOUR
pair ever measured here (pearl front/back, 0.88-0.93). Grayscale is the most duty-friendly
palette available. The caveat survives only in its narrow form: you cannot use these pieces
to argue anything about choosing COLOUR pairs for compatibility.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from PIL import Image

from shadowart import metrics as _metrics
from shadowart.geometry.projection import primary_wall_of, wall_coverage_area
from shadowart.preview.interactive3d import build_interactive
from shadowart.solve import decompose
from shadowart.targets import color as C

import face_render300 as FR
import face_paper_floor as PF
import face_duty_sweep as DS

OUT = Path("out_faces_hc/final")

FLOOR = 0.75              # GRAY_L paper -- the whole piece is material
PANEL_COUNT = 10
ANGLE_RANGE = (43, 47)    # ~45 deg: equally face-on to both walls
MIN_BOTH = 2              # HARD requirement from the brief
SEED = 2

# (label, key A, key B, density, note)
RUNS = [
    ("poe_dostoevsky_300", "poe", "dostoevsky", 1.00, "flat-mass two-tone, ~300 shards"),
    ("poe_dostoevsky_2400", "poe", "dostoevsky", 0.38, "same pair, ~2400 shards"),
    ("mona_pearl_300", "mona", "pearl", 1.00, "smooth-oil control, ~300 shards"),
]


def render_one(label, ka, kb, density, note, cands) -> dict:
    pair = (label, ka, kb)
    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label} -- {note} ===")

    old = FR.SHARD_BUDGET_PER_WALL
    FR.SHARD_BUDGET_PER_WALL = 100000            # non-binding; density sets the count
    try:
        b = PF.build_floor(pair, SEED, FLOOR, "uniform", cands,
                           panel_count=PANEL_COUNT, angle_range=ANGLE_RANGE, density=density)
    finally:
        FR.SHARD_BUDGET_PER_WALL = old

    base, rows = DS.ablate(b)
    n_both = sum(r["serves_A"] and r["serves_B"] for r in rows)
    n_dead = sum((not r["serves_A"]) and (not r["serves_B"]) for r in rows)

    print(f"  {'panel':6s}{'primary':>8s}{'dA':>10s}{'dB':>10s}{'shared':>8s}   serves")
    for r in rows:
        s = ("A" if r["serves_A"] else "") + ("B" if r["serves_B"] else "")
        print(f"  {r['panel']:6s}{r['primary']:>8s}{r['ablate_dA']:>10.5f}"
              f"{r['ablate_dB']:>10.5f}{r['shared_ratio']:>8.2f}   {s or '-':2s}"
              f"{'   <== SERVES BOTH IMAGES' if s == 'AB' else ''}")

    if n_both < MIN_BOTH:
        raise AssertionError(
            f"{label}: only {n_both} panel(s) change both walls under ablation "
            f"(need >= {MIN_BOTH}). This is not an intersection sculpture; refusing to ship it.")

    m = PF.evaluate(b, pair, SEED, FLOOR, "uniform", density=density)
    acc = _metrics.evaluate_wall_accuracy(b["targets"], base)

    print(f"  --> {n_both}/{len(rows)} panels serve BOTH images, {n_dead} dead")
    print(f"      colour-agreeing duty good={m['good_mean']:.2f}%  bad={m['bad_mean']:.2f}%"
          f"  g/b={m['good_bad']:.2f}   (best COLOUR pair on record: 0.93)")
    print(f"      shards={m['shards']}  SSIM={m['ssim']:.3f}  face_det={m['face_det']:.3f}")

    # Wall arrays are y-UP (the repo renders them with origin="lower"); flip on the way
    # out or every exported PNG is upside down.
    for w in ("A", "B"):
        Image.fromarray((np.clip(np.flipud(b["targets"][w]), 0, 1) * 255).astype(np.uint8)).save(
            out / f"target_{w}.png")
        Image.fromarray((np.clip(np.flipud(base[w]), 0, 1) * 255).astype(np.uint8)).save(
            out / f"wall_{w}.png")

    # interactive 3D: see the planes and which shards sit on them
    layout, names = b["layout"], b["names"]
    pieces = decompose.panel_stack_pieces(layout, b["stack_colorid"], names)
    poly_channel = {id(p): ch for items in pieces.values() for p, ch, _s in items}
    flat = {n: [p for p, _c, _s in items] for n, items in pieces.items()}
    build_interactive(layout, b["table"], b["opacity"], None, out / "scene_interactive.html",
                      rays=40, auto_open=False, wall_rgb=base, pieces=flat,
                      color_of=lambda panel, poly: tuple(
                          C.display_rgb(poly_channel.get(id(poly), "clear"))))

    rec = dict(m, note=note, n_dead=n_dead, min_both_required=MIN_BOTH,
               ssim_A=acc["A"]["ssim"], ssim_B=acc["B"]["ssim"],
               edge_A=acc["A"]["edge_fidelity"], edge_B=acc["B"]["edge_fidelity"],
               panel_ablation=rows,
               panel_coverage={p.name: dict(
                   area_A=wall_coverage_area(b["table"], p, "A"),
                   area_B=wall_coverage_area(b["table"], p, "B"),
                   primary=primary_wall_of(b["layout"], b["table"], p))
                   for p in b["panels"]})
    (out / "metrics.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def contact_sheet(recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2 * len(RUNS), figsize=(4.0 * len(RUNS) * 1.6, 7.6))
    for j, (label, _ka, _kb, _d, note) in enumerate(RUNS):
        r = next(x for x in recs if x["label"] == label)
        for k, w in enumerate(("A", "B")):
            c = 2 * j + k
            for i, kind in enumerate(("target", "wall")):
                ax[i, c].imshow(np.asarray(Image.open(OUT / label / f"{kind}_{w}.png")))
                ax[i, c].set_xticks([]); ax[i, c].set_yticks([])
            ax[0, c].set_title(f"{label} / wall {w}\n{note}", fontsize=8)
            ax[1, c].set_xlabel(f"{r['n_both']}/{r['n_panels']} planes serve BOTH\n"
                                f"good {r['good_mean']:.1f}%  g/b {r['good_bad']:.2f}  "
                                f"face {r['face_det']:.2f}", fontsize=7.5)
    ax[0, 0].set_ylabel("TARGET\n(GRAY_L paper)", fontsize=8)
    ax[1, 0].set_ylabel("RENDERED", fontsize=8)
    fig.suptitle("Final render -- GRAY_L paper so the whole piece is material. Every plane's "
                 "contribution verified by single-panel ablation of the real forward renderer.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "_final_compare.png", dpi=105, bbox_inches="tight")
    print(f"\nwrote {OUT}/_final_compare.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 92)
    print("FINAL RENDER -- GRAY_L paper, intersection ENFORCED "
          f"(hard assert: >= {MIN_BOTH} planes serving both images)")
    print("=" * 92)
    cands = PF.write_floor_targets(FLOOR)
    recs = [render_one(l, a, b_, d, n, cands) for l, a, b_, d, n in RUNS]
    (OUT / "summary.json").write_text(
        json.dumps([{k: v for k, v in r.items()
                     if k not in ("panel_ablation", "panel_coverage")} for r in recs],
                   indent=2), encoding="utf-8")
    contact_sheet(recs)

    print("\n" + "=" * 92)
    print(f"{'run':22s} {'shards':>7s} {'both':>7s} {'dead':>5s} {'good%':>7s} {'g/b':>6s} "
          f"{'SSIM':>6s} {'face':>6s}")
    print("-" * 92)
    for r in recs:
        print(f"{r['label']:22s} {r['shards']:>7d} {r['n_both']:>4d}/{r['n_panels']:<2d} "
              f"{r['n_dead']:>5d} {r['good_mean']:>7.2f} {r['good_bad']:>6.2f} "
              f"{r['ssim']:>6.3f} {r['face_det']:>6.3f}")
    print(f"\nwrote {OUT}/*/scene_interactive.html  (open these to see the planes)")


if __name__ == "__main__":
    main()
