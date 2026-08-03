"""Render a finished piece all the way around, not just at the stops it was built for.

Every score in `pearl3_baseline.py` is taken at the stops the optimiser was given. That is
exactly where a shadow piece flatters itself: nothing in the objective says anything about
the angles in between, and those are most of what somebody walking around the piece sees.
This renders a dense turn from the SAME panels and the SAME engraved tones, so the only
thing that changes is where the viewer is standing.

Two things come out of it:

  * a filmstrip contact sheet -- the visual gate, because a number cannot tell you that the
    figure dissolves into noise at 40 deg;
  * a continuity curve -- how fast the projection changes per degree. A piece that reads as
    one figure turning should change smoothly; one that is three unrelated images sharing a
    volume shows a spike as each stop snaps in and out.

    python pearl3_contact.py --arm 30v4 --step 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from pearl3_baseline import (ARMS, BuildConfig, _solve_once, build_scene, load_targets)
from shadowart.config.scene import Scene, TurntableSpec
from shadowart.forward.renderer import Renderer
from shadowart.geometry import turntable as TT
from shadowart.geometry.projection import build_projection_table
from shadowart.targets import color as C
from shadowart import metrics
import dataclasses


def render_sweep(cfg: BuildConfig, panel_T, angles: Sequence[float]) -> Dict[str, np.ndarray]:
    """Render the built piece at arbitrary rotations, reusing its panels and tones verbatim.

    The turntable identity is what makes this cheap: a stop is just another (wall, light)
    pair, so a denser turn is a different `TurntableSpec` over the same panels rather than a
    different piece. `panel_T` is indexed by panel, so it carries over untouched -- nothing
    is re-solved, and the images really are the same object seen from elsewhere.
    """
    dense = build_scene(cfg)
    spec = dataclasses.replace(dense.turntable,
                               stops_deg=tuple(float(a) for a in angles),
                               names=tuple(f"a{int(round(a)):03d}" for a in angles))
    walls, lights = TT.build_walls_and_lights(spec)
    scene = dataclasses.replace(dense, walls=walls, lights=lights, turntable=spec)
    table = build_projection_table(scene)
    return Renderer(scene, table).render_color_np(panel_T)


def continuity(frames: Dict[str, np.ndarray], angles: Sequence[float]) -> dict:
    """How much the projection changes from one angle to the next, per degree.

    Reported relative to the mean so it is comparable between builds. A large max/mean ratio
    means the piece lurches: it is legible at the stops and unresolved between them.
    """
    keys = list(frames)
    d = [float(np.abs(frames[keys[i + 1]] - frames[keys[i]]).mean()
               / max(angles[i + 1] - angles[i], 1e-6)) for i in range(len(keys) - 1)]
    d = np.asarray(d)
    return {"per_deg_mean": float(d.mean()), "per_deg_max": float(d.max()),
            "lurch_ratio": float(d.max() / max(d.mean(), 1e-9)),
            "per_deg": d.tolist()}


def save_filmstrip(frames: Dict[str, np.ndarray], angles, out: Path,
                   marks: Dict[float, str] | None = None, cols: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(frames)
    rows = int(np.ceil(len(keys) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.6 * rows), squeeze=False)
    for i, k in enumerate(keys):
        a = ax[i // cols][i % cols]
        a.imshow(np.clip(frames[k], 0, 1))
        a.set_xticks([]); a.set_yticks([])
        lbl = f"{angles[i]:.0f}"
        if marks and any(abs(angles[i] - m) < 1e-6 for m in marks):
            name = next(v for m, v in marks.items() if abs(angles[i] - m) < 1e-6)
            a.set_title(f"{lbl}  {name}", fontsize=7, color="crimson", fontweight="bold")
            for s in a.spines.values():
                s.set_edgecolor("crimson"); s.set_linewidth(2)
        else:
            a.set_title(lbl, fontsize=7, color="0.4")
    for j in range(len(keys), rows * cols):
        ax[j // cols][j % cols].axis("off")
    fig.suptitle("one full turn -- red frames are the angles the piece was optimised for",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), default="30v4")
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    cfg = ARMS[a.arm]
    out = Path(a.out or f"out_pearl3/contact_{a.arm}")
    out.mkdir(parents=True, exist_ok=True)

    scene = build_scene(cfg)
    table = build_projection_table(scene)
    renderer = Renderer(scene, table)
    targets = load_targets(cfg)
    names = C.palette_names(scene.color_palette)
    _, panel_T, _ = _solve_once(cfg, scene, table, renderer, names, targets)

    angles = list(np.arange(0.0, 360.0, a.step))
    # Always sample the build's own stops and the five reference angles, whatever the step,
    # so the filmstrip can be compared like for like between a 3-stop and a 5-stop build.
    from pearl3_baseline import SEQUENCE_STOPS, SEQUENCE_TARGETS
    for s in (*cfg.stops_deg, *SEQUENCE_STOPS):
        if not any(abs(s - x) < 1e-6 for x in angles):
            angles.append(float(s))
    angles = sorted(angles)
    frames = render_sweep(cfg, panel_T, angles)
    marks = {float(s): n for s, n in zip(cfg.stops_deg, cfg.views)}
    save_filmstrip(frames, angles, out / "turn.png", marks)

    # Score the three-quarter angles whether or not they were supervised. For a build that
    # only saw three stops this is the honest question: does a piece optimised at 120 deg
    # spacing happen to read at 60 deg spacing, or does the turn fall apart in between?
    unsupervised = {}
    for s, vname in zip(SEQUENCE_STOPS, SEQUENCE_TARGETS):
        if any(abs(s - x) < 1e-6 for x in cfg.stops_deg):
            continue
        key = f"a{int(round(s)):03d}"
        tgt = C.load_color_target(SEQUENCE_TARGETS[vname], (cfg.wall_res, cfg.wall_res),
                                  white_thr=0.90)
        iou, fg_p, fg_t = metrics.mark_iou(frames[key], tgt)
        unsupervised[vname] = {"angle": s, "iou": iou, "pred_fg_frac": fg_p,
                               "target_fg_frac": fg_t}

    cont = continuity(frames, angles)
    (out / "continuity.json").write_text(
        json.dumps({"arm": a.arm, "step": a.step, "continuity": cont,
                    "unsupervised_views": unsupervised}, indent=2), encoding="utf-8")

    print(f"\n=== {cfg.name}: full turn at {a.step:g} deg ===")
    print(f"continuity: {cont['per_deg_mean']:.5f} mean change per degree, "
          f"peak {cont['per_deg_max']:.5f} (lurch ratio {cont['lurch_ratio']:.2f}; "
          f"1.0 would be a perfectly even turn)")
    if unsupervised:
        print("UNSUPERVISED angles -- scored against targets the optimiser never saw:")
        for v, r in unsupervised.items():
            print(f"  {v:9s} at {r['angle']:5.0f} deg: IoU {r['iou']:.3f} "
                  f"(fg {r['pred_fg_frac']:.2f} / target {r['target_fg_frac']:.2f})")
    print(f"-> {out/'turn.png'}")


if __name__ == "__main__":
    main()
