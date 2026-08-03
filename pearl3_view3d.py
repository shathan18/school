"""Interactive 3D view of the shipped 30x30 turntable build.

Writes a self-contained HTML you can orbit in a browser: the 18 engraved sheets standing in
their three families, the three projection walls carrying the predicted image at each stop,
the lamps, and a sample of light rays.

    .venv\\Scripts\\python.exe pearl3_view3d.py                 # stage 2 (shipped tones)
    .venv\\Scripts\\python.exe pearl3_view3d.py --stage 1       # before re-toning
    .venv\\Scripts\\python.exe pearl3_view3d.py --rays 0        # no rays, lighter file
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pearl3_baseline import ARMS, run_build
from pearl3_retone import retone
from shadowart.preview.interactive3d import build_interactive
from shadowart.targets import color as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="30v4")
    ap.add_argument("--stage", type=int, default=2, choices=(1, 2))
    ap.add_argument("--sweeps", type=int, default=5)
    ap.add_argument("--rays", type=int, default=40)
    ap.add_argument("--body-only", action="store_true",
                    help="drop walls and rays; the 3 m throw otherwise shrinks the 30 cm "
                         "assembly to a tenth of the scene")
    ap.add_argument("--out", default="out_pearl3_30/v4/scene_interactive.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    cfg = ARMS[args.arm]
    res = run_build(cfg, Path("out_pearl3_30/_view3d/_scratch"))
    art = res["_artifacts"]
    scene, table, renderer = art["scene"], art["table"], art["renderer"]
    colorid, pred = art["stack_colorid"], art["pred"]

    if args.stage == 2:
        colorid, _ = retone(scene, table, renderer, art["targets"], art["names"],
                            colorid, sweeps=args.sweeps,
                            min_feature=cfg.engrave_min_feature)
        pred = renderer.render_color_np(C.stack_transmit_lut(art["names"], colorid, None))

    # colorid carries a leading stack-layer axis (one layer here); the preview indexes
    # panels directly, so drop it.
    cid = np.asarray(colorid)
    while cid.ndim > 3:
        cid = cid[0]
    opacity = (cid > 0).astype(np.float32)
    path, _ = build_interactive(
        scene, table, opacity, pred, args.out,
        rays=0 if args.body_only else args.rays,
        auto_open=not args.no_open, shard_thickness=cfg.thickness,
        panel_colorid=cid, names=art["names"],
        wall_rgb=None if args.body_only else pred,
        show_walls=not args.body_only)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"wrote {path}  ({size_mb:.1f} MB, {len(scene.panels)} sheets, "
          f"{len(scene.walls)} stops)")


if __name__ == "__main__":
    main()
