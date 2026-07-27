"""!! SUPERSEDED by `face_final_render.py` -- READ THIS FIRST !!

The fidelity numbers below are correct, but they were measured on a configuration that is
NOT an intersection sculpture, so the headline drawn from them was wrong.

This script never measured double duty at all. When it was measured afterwards
(`face_duty_check.py`, `face_crosstalk_anatomy.py`) the "winning" Poe/Dostoevsky layout had
only 2 of 14 panels changing both walls under ablation, FIVE panels changing neither, and
colour-agreeing duty of good 0.27% / bad 2.73% -- a good/bad ratio of 0.10, worse than an
ARBITRARY image pair manages (~0.3). It was two independent shadow puppets sharing a room.

The cause is the very mechanism this script celebrates. White = clear = NO SHARD, so a
two-tone poster leaves 56% of the piece as bare wall -- and bare wall cannot participate in
an intersection. Cross-talk scales with ink area (measured: ink 44% -> 2.1% of the wall
crossed; ink 85% -> 19.1%). The property that made two-tone faces "survive 300 shards" is
the same property that makes them a trivial sculpture. One finding, two signs.

Tinting the paper GRAY_L instead of leaving it clear fixes the sculpture (8-9/10 panels
serve both walls, good 16.6%, good/bad 1.66 -- roughly double the best COLOUR pair on
record) and costs the face: face_det 0.743 -> 0.378 at the same ~300 shards. More shards do
not buy it back; face_det saturates near 0.55 by ~7800 shards. Meanwhile the clear-paper
face_det is ALREADY saturated at 204 shards (0.743 -> 0.746 at 4377), which shows its
legibility was never a resolution win -- it was the absence of half the sculpture.

So: "flat mass beats smooth oil" SURVIVES (2.1x better face_det at every density).
"300 shards is enough for a face" DOES NOT survive the intersection requirement.

Kept unchanged for the record and because the fidelity harness below is reused.

--- original docstring ---

Step 3 of the face investigation: a REAL 300-shard grayscale render of the pre-test
survivors, to check whether `face_pretest.py`'s 12x12 grid proxy told the truth.

The pre-test simulates 150 shards/wall as an ideal uniform 12x12 lattice. That is an
optimistic upper bound, so it can only be wrong in one direction (real render worse).
This script removes the doubt by running the actual solver:

  * palette `noir`  (white + GRAY_L/M/D + K)   -- grayscale, per the brief
  * shard_budget 150 per wall x 2 walls        -- the 300-shard budget
  * two arms, so the theory gets its best shot:
      "uniform"    budgeted detail-biased tiling (the honest default)
      "face_dense" dual-density tiling (decompose._fragments_dual): a dense Voronoi on
                   the face box, coarse everywhere else, i.e. most of the budget spent
                   on the eyes/nose/mouth. NOTE this path does NOT self-limit to
                   `shard_budget`, so we report the ACHIEVED count and refuse to credit
                   any run that bought its quality by overspending.

  * pairs: two-tone woodcut faces (Poe / Dostoevsky) vs the smooth-oil control
    (Mona Lisa / Girl with a Pearl Earring), through an IDENTICAL pipeline.

METRIC. SSIM and edge fidelity are reported but are NOT the question -- they are
dominated by the big hair/background masses, which always reconstruct well and would
flatter a featureless blob. The question is whether the FACE lands, so the headline
number is `face_detail`: face-box-restricted detail retention of the rendered wall
against its own target, identical in form to `face_pretest.detail_retention`. Reported
next to `head_detail` (the same measure over the whole head) precisely to show the gap
between "the head reconstructs" and "the face reconstructs".

GRAYSCALE CAVEAT: hue is discarded here, so the colour-compatibility / colour-agreeing
double-duty result does not apply to any of these pairs.

Outputs -> out_faces_hc/render300/<label>_<arm>/
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from shadowart import metrics as _metrics
from shadowart.config.io import load_scene
from shadowart.forward.renderer import Renderer
from shadowart.geometry.projection import build_projection_table
from shadowart.preview.interactive3d import build_interactive
from shadowart.preview.wallview import save_color_comparison
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C

import face_pretest as FP

SCENE = "scenes/example.yaml"
OUT = Path("out_faces_hc/render300")
TGT = Path("out_faces_hc/targets")

SHARD_BUDGET_PER_WALL = FP.SHARDS_PER_WALL      # 150 -> 300 across the two walls
PANEL_COUNT = 14
ANGLE_RANGE = (30, 60)
K_CANDIDATES = 16
DAMAGE_WEIGHT = 0.5
CREDIT_WEIGHT = 1.0
MATCH_METRIC = "lab"
MATCH_TOL = 20.0
MATCH_TOL_REPORT = 25.0         # fixed reporting gate (shadowart-noise.md): never let an
                                # arm grade itself leniently with its own optimiser gate
SEEDS = [2, 7, 13]              # 3 seeds: a surprising positive must not rest on one draw

# Pairs: (label, candidate key for wall A, candidate key for wall B)
PAIRS = [
    ("twotone_poe_dostoevsky", "poe", "dostoevsky"),        # flat-mass woodcuts
    ("twotone_hatched_mallarme_wagner", "mallarme", "wagner"),  # pre-test says FAIL
    ("oilcontrol_mona_pearl", "mona", "pearl"),             # the ~2750-shard failures
]


def write_targets() -> dict[str, FP.Cand]:
    """Write the stark two-tone grayscale head crops that the solver will aim at.

    Same crop + posterise as the pre-test, so the render is answering the pre-test's own
    question rather than a slightly different one."""
    TGT.mkdir(parents=True, exist_ok=True)
    by_key = {}
    for c in FP.CANDIDATES:
        if not Path(c.path).exists():
            continue
        two = FP.posterize_gray(FP.load_head(c), 2)
        Image.fromarray((np.stack([two] * 3, -1) * 255).astype(np.uint8)).save(TGT / f"{c.key}.png")
        by_key[c.key] = c
    return by_key


def face_mask_wall(target_rgb: np.ndarray, face: tuple) -> np.ndarray:
    """Boolean wall-space mask of the face box.

    `load_color_target` crops to the non-white subject, fits it centred at
    `content_frac`, then flips so image-top -> wall-top. Rather than re-deriving that
    transform (and getting it subtly wrong), locate the subject's bounding box in the
    loaded wall array and place the face box inside it by the same fractions."""
    img_orient = np.flipud(target_rgb)                       # back to image orientation
    sub = C.subject_mask(img_orient, 0.90)
    ys, xs = np.where(sub)
    if not len(ys):
        return np.zeros(target_rgb.shape[:2], dtype=bool)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = r1 - r0 + 1, c1 - c0 + 1
    l, t, r, b = face
    m = np.zeros(target_rgb.shape[:2], dtype=bool)
    m[int(r0 + t * h):int(r0 + b * h), int(c0 + l * w):int(c0 + r * w)] = True
    return np.flipud(m)                                      # back to wall orientation


def detail_retention_wall(target_rgb: np.ndarray, pred_rgb: np.ndarray,
                          mask: np.ndarray) -> float:
    """Face-box detail retention of a rendered wall against its target -- same projection
    form as `face_pretest.detail_retention`, but over an arbitrary boolean mask."""
    from scipy import ndimage
    tg = target_rgb.mean(-1)
    pr = pred_rgb.mean(-1)
    sig = FP.HEAD_SCALE * tg.shape[1]
    a = (tg - ndimage.gaussian_filter(tg, sig))[mask]
    b = (pr - ndimage.gaussian_filter(pr, sig))[mask]
    a = a - a.mean()
    b = b - b.mean()
    den = float((a * a).sum())
    return max(0.0, float((a * b).sum()) / den) if den > 1e-9 else 0.0


def render(label: str, arm: str, ka: str, kb: str, cands: dict, seed: int) -> dict:
    out = OUT / f"{label}_{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    scene = load_scene(SCENE)
    scene = dataclasses.replace(scene, color_palette=C.PALETTES["noir"])
    names = C.palette_names(scene.color_palette)
    wr = scene.solve.wall_res

    targets = {"A": C.load_color_target(str(TGT / f"{ka}.png"), wr, white_thr=scene.white_threshold),
               "B": C.load_color_target(str(TGT / f"{kb}.png"), wr, white_thr=scene.white_threshold)}
    fmask = {"A": face_mask_wall(targets["A"], cands[ka].face),
             "B": face_mask_wall(targets["B"], cands[kb].face)}

    panels, _ = build_panels_greedy(scene, count=PANEL_COUNT, mode="deliberate",
                                    K=K_CANDIDATES, targets=targets, seed=seed,
                                    angle_deg_range=ANGLE_RANGE)
    layout = dataclasses.replace(scene, panels=panels)
    table = build_projection_table(layout)
    renderer = Renderer(layout, table)

    extra = {}
    if arm == "face_dense":
        # Give the theory its best shot: most shards onto the face box. This path skips
        # `_autotune_spacing`, so the achieved count is checked against the budget below
        # rather than assumed.
        extra = dict(face_masks=fmask, face_density=4.0, bg_coarsen=2.2)

    sc, opacity, fragments, resolved, sd, bs, si = decompose.fragment_shards_overlap(
        layout, table, targets, names=names, white_thr=layout.white_threshold,
        max_stack=layout.color_max_stack, seed=seed, shard_budget=SHARD_BUDGET_PER_WALL,
        damage_weight=DAMAGE_WEIGHT, credit_weight=CREDIT_WEIGHT,
        match_metric=MATCH_METRIC, match_tol=MATCH_TOL, **extra)

    panel_T = C.stack_transmit_lut(names, sc, si)
    pred = renderer.render_color_np(panel_T)
    acc = _metrics.evaluate_wall_accuracy(targets, pred)

    n_a = int(bs.get("A", {}).get("achieved", 0))
    n_b = int(bs.get("B", {}).get("achieved", 0))
    # "Ink area": fraction of the wall that is NOT clear-white, i.e. the only part that
    # costs shards at all. This is the mechanism to check -- a two-tone poster leaves the
    # whole lit face as bare wall (zero shards), so the entire budget lands on the dark
    # masses that carry the identity. A smooth painting is mid-tone everywhere, so every
    # pixel demands a shard and the same budget is spread thin.
    ink = {w: float(C.subject_mask(targets[w], scene.white_threshold).mean())
           for w in ("A", "B")}
    m = dict(
        label=label, arm=arm, seed=seed,
        who_A=cands[ka].who, who_B=cands[kb].who,
        group=cands[ka].group,
        ink_area_A=ink["A"], ink_area_B=ink["B"],
        shards_A=n_a, shards_B=n_b, shards_total=n_a + n_b,
        budget_per_wall=SHARD_BUDGET_PER_WALL,
        over_budget=bool(max(n_a, n_b) > SHARD_BUDGET_PER_WALL * 1.05),
        ssim_A=float(acc["A"]["ssim"]), ssim_B=float(acc["B"]["ssim"]),
        edge_A=float(acc["A"]["edge_fidelity"]), edge_B=float(acc["B"]["edge_fidelity"]),
        face_detail_A=detail_retention_wall(targets["A"], pred["A"], fmask["A"]),
        face_detail_B=detail_retention_wall(targets["B"], pred["B"], fmask["B"]),
        head_detail_A=detail_retention_wall(targets["A"], pred["A"],
                                            C.subject_mask(targets["A"], 0.90)),
        head_detail_B=detail_retention_wall(targets["B"], pred["B"],
                                            C.subject_mask(targets["B"], 0.90)),
        elapsed_s=round(time.perf_counter() - t0, 1),
    )
    flag = "  *** OVER BUDGET ***" if m["over_budget"] else ""
    print(f"  [{label} / {arm} / seed {seed}] shards {n_a}+{n_b}={n_a + n_b}{flag}"
          f"   ink area A={ink['A']:.2f} B={ink['B']:.2f}")
    print(f"      SSIM  A={m['ssim_A']:.3f} B={m['ssim_B']:.3f} | "
          f"edge A={m['edge_A']:.3f} B={m['edge_B']:.3f}")
    print(f"      head_detail A={m['head_detail_A']:.3f} B={m['head_detail_B']:.3f}   "
          f"<- big masses")
    print(f"      FACE_detail A={m['face_detail_A']:.3f} B={m['face_detail_B']:.3f}   "
          f"<- the actual question")

    save_color_comparison(targets, pred, out / "preview_final.png")
    for w in ("A", "B"):
        Image.fromarray((np.clip(np.flipud(pred[w]), 0, 1) * 255).astype(np.uint8)).save(
            out / f"wall_{w}.png")
        Image.fromarray((np.clip(np.flipud(targets[w]), 0, 1) * 255).astype(np.uint8)).save(
            out / f"target_{w}.png")

    # Interactive 3-D scene: same recipe as render_pair.py, so the shard cloud can be
    # orbited and each wall's projection inspected in a browser.
    stack_pieces = decompose.panel_stack_pieces(layout, sc, names)
    poly_channel = {id(poly): ch for items in stack_pieces.values() for poly, ch, _s in items}
    flat_pieces = {name: [poly for poly, _ch, _s in items]
                   for name, items in stack_pieces.items()}
    build_interactive(layout, table, opacity, None, out / "scene_interactive.html",
                      rays=40, auto_open=False, wall_rgb=pred, pieces=flat_pieces,
                      color_of=lambda panel, poly: tuple(
                          C.display_rgb(poly_channel.get(id(poly), "clear"))))

    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


def main() -> None:
    import sys
    # Optional filters so a single config can be regenerated without redoing the sweep:
    #   python face_render300.py [substring-of-label ...] [--arm uniform] [--seed 2]
    argv = sys.argv[1:]
    arms = ("uniform", "face_dense")
    seeds = list(SEEDS)
    if "--arm" in argv:
        i = argv.index("--arm")
        arms = (argv[i + 1],)
        del argv[i:i + 2]
    if "--seed" in argv:
        i = argv.index("--seed")
        seeds = [int(argv[i + 1])]
        del argv[i:i + 2]
    pairs = [p for p in PAIRS if not argv or any(a in p[0] for a in argv)]

    OUT.mkdir(parents=True, exist_ok=True)
    cands = write_targets()
    print("=" * 78)
    print(f"REAL {FP.TOTAL_SHARDS}-SHARD GRAYSCALE RENDER (palette=noir, "
          f"{SHARD_BUDGET_PER_WALL}/wall)")
    print("=" * 78)
    print("  GRAYSCALE: hue discarded -> the colour double-duty result does NOT apply.\n")
    rows = []
    for label, ka, kb in pairs:
        for arm in arms:
            for seed in seeds:
                rows.append(render(label, arm, ka, kb, cands, seed))
    if len(rows) == len(PAIRS) * 2 * len(SEEDS):
        (OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n" + "-" * 86)
    print(f"{'pair / arm':36s} {'shards':>12s} {'ink':>5s} {'head_det':>9s} "
          f"{'FACE_det':>15s} {'SSIM':>6s}")
    print("-" * 86)
    keys = sorted({(m["label"], m["arm"]) for m in rows})
    for label, arm in keys:
        g = [m for m in rows if m["label"] == label and m["arm"] == arm]
        sh = np.array([m["shards_total"] for m in g], float)
        ink = np.mean([0.5 * (m["ink_area_A"] + m["ink_area_B"]) for m in g])
        hd = np.array([0.5 * (m["head_detail_A"] + m["head_detail_B"]) for m in g])
        fd = np.array([0.5 * (m["face_detail_A"] + m["face_detail_B"]) for m in g])
        ss = np.mean([0.5 * (m["ssim_A"] + m["ssim_B"]) for m in g])
        over = " OVER" if any(m["over_budget"] for m in g) else ""
        print(f"{label + ' / ' + arm:36s} {sh.mean():>7.0f}+-{sh.std():<4.0f} {ink:>5.2f} "
              f"{hd.mean():>9.3f} {fd.mean():>9.3f}+-{fd.std():<4.3f} {ss:>6.3f}{over}")
    print(f"\n  wrote {OUT}/*/preview_final.png + wall_*.png + "
          f"scene_interactive.html + metrics.json")


if __name__ == "__main__":
    main()
