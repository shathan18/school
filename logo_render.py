"""Logos at ~300 shards, testing POLARITY as the intersection lever.

THE ARGUMENT, in one line: white paper = clear perspex = NO SHARD, so ink area is what
decides whether the piece can intersect at all. Measured on faces:

    ink 44% -> 2.1% of wall crossed by the other image -> 2/14 planes served both, g/b 0.10
    ink 69% -> 9.0%                                     -> 7/14
    ink 85% -> 19.1%                                    -> 8/10, g/b 1.66

A logo in its NORMAL polarity is a dark mark on white paper: ink 16-35%. That is BELOW
the case we already proved cannot intersect, so the normal arm is expected to fail. This
is not a guess -- it is the same measurement replayed on a new subject, and it is included
precisely so the failure is on the record rather than assumed.

INVERTING costs nothing. For faces the only fix we had was tinting the paper GRAY_L, which
threw away the white extreme and cost 53% of face legibility. A logo has no "correct"
polarity -- a mark reads equally well either way round -- so inversion buys the same ink
area for FREE. That is the whole reason logos are a better subject than faces here.

Three arms:
  normal          dark mark on white paper          ink 16-35%   (expected: not a sculpture)
  inverted        light mark on black ground        ink 65-76%   (expected: intersects)
  inverted_floor  GRAY_L mark on black ground       ink ~100%    (max material)

`inverted_floor` exists because plain inversion leaves the MARK itself as bare wall -- the
one region a viewer actually looks at. Tinting the mark to GRAY_L puts material back under
it. Unlike the face case this costs only the mark's brightness, not its edges.

Run:  python logo_render.py
Out:  out_logos/render/<arm>_<pair>/{scene_interactive.html,metrics.json,*.png}
      out_logos/render/_compare.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from shadowart import metrics as _metrics
from shadowart.geometry.projection import primary_wall_of, wall_coverage_area
from shadowart.preview.interactive3d import build_interactive
from shadowart.solve import decompose
from shadowart.targets import color as C

import face_pretest as FP
import face_paper_floor as PF
import face_duty_sweep as DS
import logo_pretest as LP

OUT = Path("out_logos/render")
PANEL_COUNT = 10
ANGLE_RANGE = (43, 47)      # ~45 deg: equally face-on to both walls
MIN_BOTH = 2                # HARD requirement: this must be an intersection sculpture
SEED = 2
GRAY_L = 0.75               # the lightest buildable non-clear perspex in `noir`

PAIR = ("technion5", "bgu")             # two bold flame/triangle marks, no fine text
ARMS = ["normal", "inverted", "inverted_floor"]


class Shim:
    """`PF.build_floor` wants a `face` box per subject. For a logo the whole frame IS the
    subject, and we always run the `uniform` arm, so this value is never actually used to
    steer the solver -- it only has to exist."""
    face = (0.0, 0.0, 1.0, 1.0)


def write_arm_targets(arm: str) -> dict:
    """Write both logos under the arm's polarity, named so `PF.build_floor(floor=1.0)`
    picks them up unmodified. The polarity is baked into the FILE, not into the builder."""
    PF.TGT.mkdir(parents=True, exist_ok=True)
    cands = {}
    for key in PAIR:
        two = FP.posterize_gray(LP.load_square(key, 0.04), 2)
        if arm == "normal":
            img = two
        elif arm == "inverted":
            img = 1.0 - two
        elif arm == "inverted_floor":
            # invert, then pull the light tone down to GRAY_L so the MARK carries material
            inv = 1.0 - two
            img = np.where(inv > 0.5, GRAY_L, 0.0)
        else:
            raise ValueError(arm)
        name = f"{key}_{arm}"
        Image.fromarray((np.clip(np.stack([img] * 3, -1), 0, 1) * 255).astype(np.uint8)).save(
            PF.TGT / f"{name}_f100.png")
        cands[name] = Shim()
    return cands


def mark_iou(target_rgb: np.ndarray, wall_rgb: np.ndarray) -> float:
    """Shape fidelity of the MARK, which is what a logo is judged on.

    SSIM punishes tonal drift across the whole frame; for a 2-tone mark the only question
    is whether the silhouette landed. Threshold both at the target's mid-level and take
    IoU of the lighter region."""
    t = target_rgb.mean(-1)
    r = wall_rgb.mean(-1)
    thr = 0.5 * (float(t.min()) + float(t.max()))
    a, b = t > thr, r > thr
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def render_one(arm: str) -> dict:
    ka, kb = f"{PAIR[0]}_{arm}", f"{PAIR[1]}_{arm}"
    label = f"{arm}"
    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {label} ===")

    cands = write_arm_targets(arm)
    b = PF.build_floor((label, ka, kb), SEED, 1.0, "uniform", cands,
                       panel_count=PANEL_COUNT, angle_range=ANGLE_RANGE)

    base, rows = DS.ablate(b)
    n_both = sum(r["serves_A"] and r["serves_B"] for r in rows)
    n_dead = sum((not r["serves_A"]) and (not r["serves_B"]) for r in rows)

    print(f"  {'panel':6s}{'primary':>8s}{'dA':>10s}{'dB':>10s}   serves")
    for r in rows:
        s = ("A" if r["serves_A"] else "") + ("B" if r["serves_B"] else "")
        print(f"  {r['panel']:6s}{r['primary']:>8s}{r['ablate_dA']:>10.5f}"
              f"{r['ablate_dB']:>10.5f}   {s or '-':2s}"
              f"{'   <== SERVES BOTH IMAGES' if s == 'AB' else ''}")

    b["_faceA"], b["_faceB"] = Shim.face, Shim.face
    m = PF.evaluate(b, (label, ka, kb), SEED, 1.0, "uniform")
    acc = _metrics.evaluate_wall_accuracy(b["targets"], base)
    ink = {w: float((b["targets"][w].mean(-1) < b["layout"].white_threshold).mean() * 100)
           for w in ("A", "B")}
    iou = {w: mark_iou(b["targets"][w], base[w]) for w in ("A", "B")}
    iou_mean = float(np.mean(list(iou.values())))

    verdict = "INTERSECTS" if n_both >= MIN_BOTH else "NOT A SCULPTURE"
    print(f"  --> {n_both}/{len(rows)} planes serve BOTH images, {n_dead} dead   [{verdict}]")
    print(f"      ink A/B = {ink['A']:.1f}% / {ink['B']:.1f}%")
    print(f"      duty good={m['good_mean']:.2f}%  bad={m['bad_mean']:.2f}%  g/b={m['good_bad']:.2f}")
    print(f"      shards={m['shards']}  SSIM={m['ssim']:.3f}  mark_IoU={iou_mean:.3f}")

    for w in ("A", "B"):
        # wall arrays are y-UP; flip or every exported PNG is upside down
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

    rec = dict(m, arm=arm, pair=list(PAIR), n_both=n_both, n_dead=n_dead,
               n_panels=len(rows), verdict=verdict, ink=ink,
               mark_iou=iou, mark_iou_mean=iou_mean,
               ssim_A=acc["A"]["ssim"], ssim_B=acc["B"]["ssim"],
               panel_ablation=rows,
               panel_coverage={p.name: dict(
                   area_A=wall_coverage_area(b["table"], p, "A"),
                   area_B=wall_coverage_area(b["table"], p, "B"),
                   primary=primary_wall_of(b["layout"], b["table"], p))
                   for p in b["panels"]})
    (out / "metrics.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def contact_sheet(recs) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2 * len(recs), figsize=(3.2 * len(recs) * 1.7, 7.4))
    for j, r in enumerate(recs):
        for k, w in enumerate(("A", "B")):
            c = 2 * j + k
            for i, kind in enumerate(("target", "wall")):
                ax[i, c].imshow(np.asarray(Image.open(OUT / r["arm"] / f"{kind}_{w}.png")))
                ax[i, c].set_xticks([]); ax[i, c].set_yticks([])
            ax[0, c].set_title(f"{r['arm']} / wall {w}\nink {r['ink'][w]:.0f}%", fontsize=8)
            ax[1, c].set_xlabel(f"{r['n_both']}/{r['n_panels']} planes serve BOTH\n"
                                f"g/b {r['good_bad']:.2f}   IoU {r['mark_iou'][w]:.2f}\n"
                                f"{r['verdict']}", fontsize=7.5)
    ax[0, 0].set_ylabel("TARGET", fontsize=9)
    ax[1, 0].set_ylabel("RENDERED", fontsize=9)
    fig.suptitle("Polarity is the intersection lever. White paper = clear perspex = no shard, "
                 "so a normal logo has too little material to intersect.\n"
                 "Every plane's contribution verified by single-panel ablation of the real "
                 "forward renderer.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "_compare.png", dpi=105, bbox_inches="tight")
    print(f"\nwrote {OUT}/_compare.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 94)
    print(f"LOGOS @ ~300 shards -- polarity sweep, pair = {PAIR[0]} x {PAIR[1]}")
    print("=" * 94)
    recs = [render_one(a) for a in ARMS]
    (OUT / "summary.json").write_text(
        json.dumps([{k: v for k, v in r.items()
                     if k not in ("panel_ablation", "panel_coverage")} for r in recs],
                   indent=2), encoding="utf-8")
    contact_sheet(recs)

    print("\n" + "=" * 94)
    print(f"{'arm':16s}{'inkA%':>7s}{'inkB%':>7s}{'shards':>8s}{'both':>7s}{'dead':>6s}"
          f"{'good%':>8s}{'g/b':>7s}{'SSIM':>7s}{'IoU':>7s}   verdict")
    print("-" * 94)
    for r in recs:
        print(f"{r['arm']:16s}{r['ink']['A']:>7.1f}{r['ink']['B']:>7.1f}{r['shards']:>8d}"
              f"{r['n_both']:>4d}/{r['n_panels']:<2d}{r['n_dead']:>6d}"
              f"{r['good_mean']:>8.2f}{r['good_bad']:>7.2f}{r['ssim']:>7.3f}"
              f"{r['mark_iou_mean']:>7.3f}   {r['verdict']}")
    print(f"\nwrote {OUT}/*/scene_interactive.html  (open these to see the planes)")


if __name__ == "__main__":
    main()
