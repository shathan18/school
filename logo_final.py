"""FINAL: two logo pieces where the planes do genuinely SHARED work.

The problem with `logo_render.py --inverted`: it satisfied the letter of the brief (6/10
planes verified moving both walls) but not the spirit. Its colour-agreeing duty was
good 0.89% / bad 10.50%, g/b 0.08 -- i.e. the planes touch both walls, but almost
everything the second wall receives is NOISE. A plane that ruins the other image is
"contributing to both" only in a lawyer's sense.

Two solver levers exist for this and NOTHING in the repo has ever used either:

  colour_blend  -- today a shard's colour is fixed from its own wall BEFORE its host plane
                   is chosen, so it can only help the other wall by luck. With
                   colour_blend > 0 the (host, colour) pair is chosen TOGETHER: for each
                   candidate plane the solver looks at what the other wall wants where the
                   shard would land and blends toward it, accepting only compromises that
                   stay within `colour_primary_tol` of what its own wall wanted. Under
                   `noir` both walls want the same four neutral greys, so the compromise is
                   nearly free -- this is the palette where the lever should pay best.

  joint_prior   -- per-panel opacity from the two-wall JOINT optimiser (optimizer.solve,
                   Adam on the real differentiable renderer), warped to each wall. Adding
                   it to the host objective pulls each shard onto the plane the joint
                   solution put material on, i.e. a depth chosen to exploit cross-talk,
                   instead of the per-wall "dodge the other image" heuristic.

Two pieces, deliberately different so the result is not one lucky pair:
  piece1  technion5 x bgu   -- two bold marks, LIGHT-on-black (mark stays bare wall)
  piece2  technion  x huji  -- GRAY_L-on-black, so the mark itself carries material too

Run:  python logo_final.py
Out:  out_logos/final/<piece>/{scene_interactive.html, metrics.json, *.png}
      out_logos/final/_final.png
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from shadowart import metrics as _metrics
from shadowart.forward.backend import DTYPE, DEVICE, to_t, to_np
from shadowart.geometry.projection import primary_wall_of, wall_coverage_area
from shadowart.preview.interactive3d import build_interactive
from shadowart.solve import decompose, optimizer
from shadowart.targets import color as C

import face_pretest as FP
import face_paper_floor as PF
import face_duty_sweep as DS
import logo_pretest as LP
import logo_render as LR

OUT = Path("out_logos/final")
PANEL_COUNT = 10
ANGLE_RANGE = (43, 47)
MIN_BOTH = 2
SEED = 2
GRAY_L = 0.75
GRAY_M = 0.53
GRAY_D = 0.34
MARGIN = 0.12          # white border around the mark, as a fraction of its long side

# (label, keyA, keyB, arm, density, joint_weight, colour_blend, note)
PIECES = [
    ("piece1_technion_bgu", "technion", "bgu", "greymark_m", 1.00, 0.35, 0.35,
     "official shield vs the flame -- the strongest shared-duty pair of the set"),
    ("piece2_cs_technion", "cs", "technion", "greymark_m", 0.70, 0.35, 0.35,
     "CS faculty mark vs the Technion shield; CS is a thin 12%-ink mark, so it needs "
     "finer shards (d=0.70) and it must sit on wall A to reach 8/10"),
]


def write_targets(keys, arm) -> dict:
    """Polarity/tone transforms. The `midtone` arm is the one that matters.

    `colour_agreeing_duty` renders the wall with its OWN panels removed and asks whether the
    remaining cross-talk landed in a colour that wall wants, within match_tol=0.30 (RGB L2).
    A single sheet only reaches K=0.20, and ||(0.20,0.20,0.20)-(0,0,0)|| = 0.35 > 0.30, so a
    PURE BLACK subject pixel can never be scored good -- it needs two stacked K sheets, which
    is by definition not single-plane double duty. Every earlier arm grounded the image in
    black, so their g/b was capped by the palette, not by the layout.

    `midtone` puts both tones where ONE sheet lands exactly: GRAY_L 0.75 and GRAY_D 0.34.
    Now a single plane can simultaneously be the correct tone for both walls, which is the
    physical condition for a real intersection rather than a geometric coincidence."""
    PF.TGT.mkdir(parents=True, exist_ok=True)
    cands = {}
    for key in keys:
        two = FP.posterize_gray(LP.load_square(key, MARGIN), 2)
        if arm == "normal":
            img = two
        elif arm == "inverted":
            img = 1.0 - two
        elif arm == "inverted_floor":
            inv = 1.0 - two
            img = np.where(inv > 0.5, GRAY_L, 0.0)
        elif arm == "midtone":
            inv = 1.0 - two
            img = np.where(inv > 0.5, GRAY_L, GRAY_D)
        elif arm.startswith("greymark"):
            # Mark DARK on a WHITE ground -- but the mark is a tone ONE sheet can hit.
            # White ground is above white_thr so it is not subject and never scores; it is
            # also bare wall, which is what kept the `normal` arm legible (IoU 0.93). The
            # only change from `normal` is that the mark is GRAY_D/GRAY_M instead of black,
            # which is precisely the change that lets cross-talk landing on it be correct.
            tone = GRAY_M if arm.endswith("_m") else GRAY_D
            img = np.where(two > 0.5, 1.0, tone)
        else:
            raise ValueError(arm)
        name = f"{key}_{arm}"
        Image.fromarray((np.clip(np.stack([img] * 3, -1), 0, 1) * 255).astype(np.uint8)).save(
            PF.TGT / f"{name}_f100.png")
        cands[name] = LR.Shim()
    return cands


def joint_prior(scene, renderer, targets_rgb) -> dict:
    """Run the JOINT two-wall optimiser, then warp each panel's opacity onto each wall.

    `optimizer.solve` works on DARKNESS maps, while our targets are RGB, so convert.
    The result is [P,Hn,Wn] per wall, indexed by GLOBAL panel index -- which is what
    `fragment_shards_overlap` indexes it with (`jpf[fam[k][0]]`)."""
    dark = {w: (1.0 - targets_rgb[w].mean(-1)).astype(np.float32) for w in ("A", "B")}
    op, _hist = optimizer.solve(scene, renderer, dark, verbose=False)   # [P,Hp,Wp]
    P, Hp, Wp = op.shape
    prior = {}
    for wall in scene.walls:
        grids = renderer._grids[wall]
        arr = np.zeros((P, renderer.Hn, renderer.Wn), np.float32)
        for pi in range(P):
            t = to_t(op[pi]).view(1, 1, Hp, Wp)
            c = F.grid_sample(t, grids[pi], mode="bilinear",
                              padding_mode="zeros", align_corners=False)
            arr[pi] = to_np(c[0, 0])
        prior[wall] = arr
    return prior


def build(label, ka, kb, arm, density, jw, cb, cands, use_levers: bool):
    """`use_levers=False` reproduces the previous behaviour -- the honest control."""
    extra = {}
    if use_levers:
        # The prior needs a renderer, which needs the panels -- so build once WITHOUT the
        # prior to get the geometry, compute the prior on that geometry, then rebuild.
        # Panel choice is seeded and independent of the prior, so the geometry is identical.
        b0 = PF.build_floor((label, ka, kb), SEED, 1.0, "uniform", cands,
                            panel_count=PANEL_COUNT, angle_range=ANGLE_RANGE,
                            density=density,
                            target_kw=dict(crop_mode="all", content_frac=0.88))
        extra = dict(joint_prior=joint_prior(b0["layout"], b0["renderer"], b0["targets"]),
                     joint_weight=jw, colour_blend=cb, colour_primary_tol=0.16)
    return PF.build_floor((label, ka, kb), SEED, 1.0, "uniform", cands,
                          panel_count=PANEL_COUNT, angle_range=ANGLE_RANGE,
                          density=density, extra_kw=extra,
                          target_kw=dict(crop_mode="all", content_frac=0.88))


def score(b, label, ka, kb, density):
    base, rows = DS.ablate(b)
    n_both = sum(r["serves_A"] and r["serves_B"] for r in rows)
    n_dead = sum((not r["serves_A"]) and (not r["serves_B"]) for r in rows)
    b["_faceA"], b["_faceB"] = LR.Shim.face, LR.Shim.face
    m = PF.evaluate(b, (label, ka, kb), SEED, 1.0, "uniform", density=density)
    acc = _metrics.evaluate_wall_accuracy(b["targets"], base)
    iou = {w: LR.mark_iou(b["targets"][w], base[w]) for w in ("A", "B")}
    shared = float(np.mean([r["shared_ratio"] for r in rows]))
    return base, rows, dict(
        n_both=n_both, n_dead=n_dead, n_panels=len(rows), shards=m["shards"],
        good=m["good_mean"], bad=m["bad_mean"], gb=m["good_bad"],
        ssim=float(np.mean([acc["A"]["ssim"], acc["B"]["ssim"]])),
        iou_A=iou["A"], iou_B=iou["B"], iou=float(np.mean(list(iou.values()))),
        shared_ratio=shared)


def run_piece(label, ka, kb, arm, density, jw, cb, note) -> dict:
    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 92}\n=== {label}  ({ka} x {kb}, {arm}) -- {note}\n{'=' * 92}")
    cands = write_targets((ka, kb), arm)
    nka, nkb = f"{ka}_{arm}", f"{kb}_{arm}"

    print("  [shipped] tonal design only ...")
    b = build(label, nka, nkb, arm, density, jw, cb, cands, use_levers=False)
    base, rows, s = score(b, label, nka, nkb, density)
    print(f"    both={s['n_both']}/{s['n_panels']}  good={s['good']:.2f}%  "
          f"bad={s['bad']:.2f}%  g/b={s['gb']:.2f}  IoU={s['iou']:.3f}")

    # Ablation, NOT the shipped piece: the two unused solver levers. They are reported
    # because they were the obvious idea and they did not reliably pay -- see the summary.
    print(f"  [ablation] joint_prior w={jw}, colour_blend={cb} ...")
    b_lev = build(label, nka, nkb, arm, density, jw, cb, cands, use_levers=True)
    _, _, s_lev = score(b_lev, label, nka, nkb, density)
    print(f"    both={s_lev['n_both']}/{s_lev['n_panels']}  good={s_lev['good']:.2f}%  "
          f"bad={s_lev['bad']:.2f}%  g/b={s_lev['gb']:.2f}  IoU={s_lev['iou']:.3f}")

    print(f"\n  {'panel':6s}{'primary':>8s}{'dA':>10s}{'dB':>10s}{'shared':>8s}   serves")
    for r in rows:
        t = ("A" if r["serves_A"] else "") + ("B" if r["serves_B"] else "")
        print(f"  {r['panel']:6s}{r['primary']:>8s}{r['ablate_dA']:>10.5f}"
              f"{r['ablate_dB']:>10.5f}{r['shared_ratio']:>8.2f}   {t or '-':2s}"
              f"{'   <== SERVES BOTH IMAGES' if t == 'AB' else ''}")

    if s["n_both"] < MIN_BOTH:
        raise AssertionError(f"{label}: only {s['n_both']} plane(s) move both walls "
                             f"(need >= {MIN_BOTH}). Refusing to ship it.")

    for w in ("A", "B"):
        Image.fromarray((np.clip(np.flipud(b["targets"][w]), 0, 1) * 255).astype(np.uint8)) \
            .save(out / f"target_{w}.png")
        Image.fromarray((np.clip(np.flipud(base[w]), 0, 1) * 255).astype(np.uint8)) \
            .save(out / f"wall_{w}.png")

    layout, names = b["layout"], b["names"]
    pieces = decompose.panel_stack_pieces(layout, b["stack_colorid"], names)
    poly_channel = {id(p): ch for items in pieces.values() for p, ch, _s in items}
    flat = {n: [p for p, _c, _s in items] for n, items in pieces.items()}
    build_interactive(layout, b["table"], b["opacity"], None, out / "scene_interactive.html",
                      rays=40, auto_open=False, wall_rgb=base, pieces=flat,
                      color_of=lambda panel, poly: tuple(
                          C.display_rgb(poly_channel.get(id(poly), "clear"))))

    rec = dict(label=label, pair=[ka, kb], arm=arm, density=density, note=note,
               joint_weight=jw, colour_blend=cb, shipped=s, lever_ablation=s_lev,
               panel_ablation=rows,
               panel_coverage={p.name: dict(
                   area_A=wall_coverage_area(b["table"], p, "A"),
                   area_B=wall_coverage_area(b["table"], p, "B"),
                   primary=primary_wall_of(b["layout"], b["table"], p))
                   for p in b["panels"]})
    (out / "metrics.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def sheet(recs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(13.2, 7.4))
    for j, r in enumerate(recs):
        for k, w in enumerate(("A", "B")):
            c = 2 * j + k
            for i, kind in enumerate(("target", "wall")):
                ax[i, c].imshow(np.asarray(Image.open(OUT / r["label"] / f"{kind}_{w}.png")))
                ax[i, c].set_xticks([]); ax[i, c].set_yticks([])
            ax[0, c].set_title(f"{r['label'].split('_', 1)[1]} / wall {w}", fontsize=8)
        s, c0 = r["shipped"], r["lever_ablation"]
        ax[1, 2 * j].set_xlabel(
            f"{s['n_both']}/{s['n_panels']} planes serve BOTH  ({s['n_dead']} dead)\n"
            f"good {s['good']:.1f}%  bad {s['bad']:.1f}%  g/b {s['gb']:.1f}",
            fontsize=7.5)
        ax[1, 2 * j + 1].set_xlabel(
            f"IoU {s['iou_A']:.2f} / {s['iou_B']:.2f}   {s['shards']} shards",
            fontsize=7.5)
        # the note is prose and can be long -- give it the full width of the piece, wrapped,
        # instead of letting it run into the neighbouring piece's caption
        fig.text(0.28 + 0.5 * j, 0.005, "\n".join(textwrap.wrap(r["note"], 62)),
                 ha="center", va="bottom", fontsize=7.5)
    ax[0, 0].set_ylabel("TARGET", fontsize=9)
    ax[1, 0].set_ylabel("RENDERED", fontsize=9)
    fig.suptitle("Two pieces, one tonal rule: the mark is GRAY_M -- a tone a SINGLE sheet hits "
                 "exactly -- on bare wall.\nThat is what lets one plane be the correct tone for "
                 "both images at once; every plane below is ablation-verified on the real "
                 "forward renderer.", fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.11)
    fig.savefig(OUT / "_final.png", dpi=110)
    print(f"\nwrote {OUT}/_final.png")


def check_targets() -> None:
    """Look at the framing BEFORE spending a render on it -- the face campaign lost a whole
    round to hand-set crops that were never actually looked at."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["technion", "technion5", "cs", "huji", "tau", "bgu"]
    fig, ax = plt.subplots(2, len(keys), figsize=(2.5 * len(keys), 5.4))
    for j, k in enumerate(keys):
        two = FP.posterize_gray(LP.load_square(k, MARGIN), 2)
        shown = np.where(two > 0.5, 1.0, GRAY_M)
        for i, img in enumerate((two, shown)):
            ax[i, j].imshow(img, cmap="gray", vmin=0, vmax=1)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            for f in (0.0, 1.0):                      # wall edge, so clipping is obvious
                ax[i, j].axhline(f * (img.shape[0] - 1), color="r", lw=1.2)
                ax[i, j].axvline(f * (img.shape[1] - 1), color="r", lw=1.2)
        ax[0, j].set_title(k, fontsize=9)
    ax[0, 0].set_ylabel("2-tone", fontsize=9)
    ax[1, 0].set_ylabel(f"greymark_m", fontsize=9)
    fig.suptitle(f"Target framing at MARGIN={MARGIN}: the whole mark must sit INSIDE the "
                 f"red wall edge.", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "_targets.png", dpi=110, bbox_inches="tight")
    print(f"wrote {OUT}/_targets.png")


def scan_pairs() -> None:
    """Choose the second pair on ink area + measured duty, not on taste.

    Ink area is the whole game (see out_logos ink table): a thin mark leaves too little
    material for a plane to be shared, exactly as a low-ink face did. HUJI's menorah is
    the thinnest mark in the set and it is expected to fail here.
    """
    arm, dens = "greymark_m", 1.00
    keys = ["technion", "technion5", "cs", "huji", "tau", "bgu"]
    ink = {k: 100.0 * float((np.where(FP.posterize_gray(LP.load_square(k, MARGIN), 2) > 0.5,
                                      1.0, GRAY_M) < 0.90).mean()) for k in keys}
    print("ink area at the shipped framing:")
    for k in keys:
        print(f"  {k:12s}{ink[k]:6.1f}%")
    pairs = [("technion", "bgu"), ("technion", "cs"), ("technion5", "cs"),
             ("cs", "bgu"), ("cs", "tau"), ("technion5", "tau")]
    print(f"\n{'pair':26s}{'inkA/B':>12s}{'both':>7s}{'good%':>8s}{'g/b':>8s}"
          f"{'IoU':>7s}{'SSIM':>7s}{'shards':>8s}")
    print("-" * 84)
    for ka, kb in pairs:
        cands = write_targets((ka, kb), arm)
        nka, nkb = f"{ka}_{arm}", f"{kb}_{arm}"
        b = build("scan", nka, nkb, arm, dens, 0.0, 0.0, cands, use_levers=False)
        _, _, s = score(b, "scan", nka, nkb, dens)
        print(f"{ka + ' x ' + kb:26s}{ink[ka]:>5.0f}/{ink[kb]:<6.0f}"
              f"{s['n_both']:>4d}/{s['n_panels']:<2d}{s['good']:>8.2f}{s['gb']:>8.2f}"
              f"{s['iou']:>7.3f}{s['ssim']:>7.3f}{s['shards']:>8d}")


def sweep_arms() -> None:
    """Pick the tonal design on evidence instead of taste: legibility AND duty, same pair."""
    label, ka, kb = "sweep", "technion5", "bgu"
    print(f"{'arm':16s}{'dens':>6s}{'both':>7s}{'good%':>8s}{'bad%':>8s}{'g/b':>7s}"
          f"{'IoU':>7s}{'SSIM':>7s}{'shards':>8s}")
    print("-" * 74)
    for arm in ("normal", "greymark", "greymark_m", "midtone"):
        for dens in (1.0, 0.60):
            cands = write_targets((ka, kb), arm)
            nka, nkb = f"{ka}_{arm}", f"{kb}_{arm}"
            b = build(label, nka, nkb, arm, dens, 0.0, 0.0, cands, use_levers=False)
            _, _, s = score(b, label, nka, nkb, dens)
            print(f"{arm:16s}{dens:>6.2f}{s['n_both']:>4d}/{s['n_panels']:<2d}"
                  f"{s['good']:>8.2f}{s['bad']:>8.2f}{s['gb']:>7.2f}{s['iou']:>7.3f}"
                  f"{s['ssim']:>7.3f}{s['shards']:>8d}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [run_piece(*p) for p in PIECES]
    (OUT / "summary.json").write_text(
        json.dumps([{k: v for k, v in r.items()
                     if k not in ("panel_ablation", "panel_coverage")} for r in recs],
                   indent=2), encoding="utf-8")
    sheet(recs)

    print("\n" + "=" * 92)
    print(f"{'piece':24s}{'arm':16s}{'':>4s}{'both':>7s}{'good%':>8s}{'bad%':>8s}"
          f"{'g/b':>7s}{'IoU':>7s}{'shards':>8s}")
    print("-" * 92)
    for r in recs:
        for tag, s in (("SHIPPED", r["shipped"]), ("levers", r["lever_ablation"])):
            print(f"{r['label']:24s}{r['arm']:16s}{tag:>8s}"
                  f"{s['n_both']:>3d}/{s['n_panels']:<3d}{s['good']:>8.2f}{s['bad']:>8.2f}"
                  f"{s['gb']:>7.2f}{s['iou']:>7.3f}{s['shards']:>8d}")
    print(f"\nwrote {OUT}/*/scene_interactive.html")


if __name__ == "__main__":
    import sys
    if "--pairs" in sys.argv:
        scan_pairs()
    elif "--targets" in sys.argv:
        check_targets()
    elif "--sweep" in sys.argv:
        sweep_arms()
    else:
        main()
