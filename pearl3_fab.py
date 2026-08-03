"""Turn a solved build into fabricable files, and then refuse to trust them.

Three gates run after the export, because every one of them has caught a real failure in
this repo before:

GATE A -- weave feasibility. Three sheet families at 60 degrees is a triangular grid, and
    triangular grids famously want to meet three-at-a-point. A triple point cannot be
    half-lapped: each sheet would have to be simultaneously notched from the top and from
    the bottom at the same u. This measures the multiplicity of every crossing cluster and
    the widest slot, and fails loudly rather than shipping a model that cannot assemble.

GATE B -- fabrication round trip. The raster score is not the score of the object. The
    pipeline threshold -> min-feature -> contour -> kerf changes the shapes, and engraving
    forces one laser power per region where the solver was free to vary intensity pixel by
    pixel. So the vector output is rasterised BACK (shapely point-in-polygon on pixel
    CENTRES -- a scanline refill of marching-squares contours was measured to lose 6 points
    of IoU purely as an artefact) and re-rendered. The gap between the two numbers is the
    honest cost of making the thing.

GATE C -- single-panel ablation. Duty statistics are computed from the solver's own
    bookkeeping and can agree with themselves while being wrong. Deleting one sheet and
    re-rendering cannot be faked: if removing a sheet changes nothing, that sheet is
    decoration.

Run:  .venv\\Scripts\\python.exe pearl3_fab.py --arm 30v4 --out out_pearl3_30
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

import pearl3_baseline as P
import pearl3_retone as R
from shadowart import metrics
from shadowart.fabricate import export_engrave, export_obj
from shadowart.fabricate.joints import build_panel_drawings
from shadowart.geometry.panels import weave_crossings
from shadowart.solve import decompose
from shadowart.targets import color as C


# --- gate A: can it be woven? ---------------------------------------------------------
def check_weave(scene, tol=2e-3):
    """Cluster crossings by floor-plan position; any cluster with >2 sheets is unbuildable.

    `tol` is in metres and is deliberately larger than the slot width: two crossings a
    fraction of a millimetre apart are not two joints, they are one impossible joint with a
    sliver of material between them that will snap on assembly.
    """
    cr = weave_crossings(scene)
    clusters = []                       # [(x, y, {panel names})]
    for c in cr:
        for cl in clusters:
            if abs(cl[0] - c.x) < tol and abs(cl[1] - c.y) < tol:
                cl[2].update((c.panelA_name, c.panelB_name))
                break
        else:
            clusters.append([c.x, c.y, {c.panelA_name, c.panelB_name}])
    mult = [len(cl[2]) for cl in clusters] or [0]
    worst = max(clusters, key=lambda cl: len(cl[2])) if clusters else None
    return {
        "n_crossings": len(cr),
        "n_clusters": len(clusters),
        "max_multiplicity": int(max(mult)),
        "n_triple_points": int(sum(1 for m in mult if m > 2)),
        "worst_cluster": (None if worst is None else
                          {"xy_m": [round(worst[0], 4), round(worst[1], 4)],
                           "panels": sorted(worst[2])}),
        "min_crossing_angle_deg": round(min((c.angle_deg for c in cr), default=90.0), 2),
        "max_slot_width_mm": round(max((c.slot_width for c in cr), default=0.0) * 1000, 3),
        "material_thickness_mm": scene.material_thickness * 1000,
        "pass": bool(max(mult) <= 2),
    }


# --- gate B: does the cut file still make the picture? ---------------------------------
def rasterise_pieces(scene, pieces_by_panel, names, stack_intensity, shape):
    """Vector polygons -> (stack_colorid, stack_intensity) by testing pixel CENTRES.

    Intensity is collapsed to ONE value per polygon (the mean of the solver's intensities
    inside it). That is not an approximation for convenience -- it is the fabrication
    constraint. A raster engrave pass fires at a single power over a region; the solver's
    freedom to shade a shard's interior pixel by pixel does not survive contact with the
    machine, and this gate exists to price exactly that kind of loss.
    """
    from shapely import contains_xy
    S, G, Hp, Wp = shape
    cid = np.zeros((S, G, Hp, Wp), dtype=np.int16)
    inten = np.zeros((S, G, Hp, Wp), dtype=np.float32)
    idx_of = {n: i for i, n in enumerate(names)}
    for gi, panel in enumerate(scene.panels):
        (u0, u1), (v0, v1) = panel.u_range, panel.v_range
        uu = u0 + (np.arange(Wp) + 0.5) * (u1 - u0) / Wp        # pixel centres, not corners
        vv = v0 + (np.arange(Hp) + 0.5) * (v1 - v0) / Hp
        UU, VV = np.meshgrid(uu, vv)
        for poly, channel, s in pieces_by_panel.get(panel.name, []):
            k = idx_of.get(channel)
            if k is None:
                continue
            m = contains_xy(poly, UU, VV)
            if not m.any():
                continue
            src = stack_intensity[s, gi][m]
            lit = src[src > 0]
            cid[s, gi][m] = k
            inten[s, gi][m] = float(lit.mean()) if lit.size else 1.0
    return cid, inten


def fab_round_trip(scene, table, renderer, targets, names, pieces_by_panel, stack_intensity):
    cid, inten = rasterise_pieces(scene, pieces_by_panel, names, stack_intensity,
                                  stack_intensity.shape)
    panel_T = C.stack_transmit_lut(names, cid, inten)
    pred = renderer.render_color_np(panel_T)
    return metrics.evaluate_multiview(targets, pred), pred


# --- gate C: is every sheet doing work? ------------------------------------------------
def ablate_panels(scene, renderer, targets, panel_T):
    """Blank one sheet at a time; report the IoU each view loses. The unfakeable duty test."""
    base = metrics.evaluate_multiview(targets, renderer.render_color_np(panel_T))
    walls = list(scene.walls)
    rows = []
    for gi, panel in enumerate(scene.panels):
        q = panel_T.copy()
        q[gi] = 1.0                              # clear sheet: present, but blank
        ev = metrics.evaluate_multiview(targets, renderer.render_color_np(q))
        drop = {w: base[w]["iou"] - ev[w]["iou"] for w in walls}
        rows.append({"panel": panel.name,
                     "drop_by_view": {w: round(d, 4) for w, d in drop.items()},
                     "best_view": max(drop, key=drop.get),
                     "max_drop": round(max(drop.values()), 4),
                     "mean_iou_drop": round(base["_summary"]["mean_iou"]
                                            - ev["_summary"]["mean_iou"], 4)})
    idle = [r["panel"] for r in rows if r["max_drop"] < 0.002]
    return {"per_panel": rows, "idle_panels": idle, "n_idle": len(idle),
            "pass": len(idle) == 0}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="30v4", choices=sorted(P.ARMS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-obj", action="store_true")
    ap.add_argument("--retone", type=int, default=3,
                    help="stage-2 tone coordinate-descent sweeps (0 = ship stage 1 as solved)")
    a = ap.parse_args(argv)

    out = Path(a.out or f"out_pearl3_30/{a.arm}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = P.ARMS[a.arm]

    t0 = time.perf_counter()
    res = P.run_build(cfg, out, verbose=True)
    art = res.pop("_artifacts")
    scene, table, renderer = art["scene"], art["table"], art["renderer"]
    names, targets = art["names"], art["targets"]

    colorid, panel_T, retone_log = art["stack_colorid"], art["panel_T"], None
    live = metrics.evaluate_multiview(targets, art["pred"])
    if a.retone > 0:
        print("\nstage 2: re-toning with the geometry frozen")
        colorid, retone_log = R.retone(scene, table, renderer, targets, names,
                                       art["stack_colorid"], sweeps=a.retone,
                                       min_feature=cfg.engrave_min_feature)
        panel_T = C.stack_transmit_lut(names, colorid, None)
        m2 = metrics.evaluate_multiview(targets, renderer.render_color_np(panel_T))
        ss, rm = R._tone_stats(m2, list(scene.walls))
        print(metrics.format_multiview_report(m2, f"{a.arm} re-toned -- THIS is what ships"))
        print(f"  SSIM {ss:.4f}  RMSE {rm:.4f}")
        res["summary"] = m2["_summary"]
        res["views"] = {w: m2[w] for w in scene.walls}
        live = m2

    print("\nvectorising shards (threshold -> min-feature -> contours -> kerf) ...")
    # Engrave regions are vectorised against the BEAM SPOT, not the structural cut limit:
    # `min_feature` exists so a shard cut OUT of a sheet survives handling, and an engraved
    # region is never cut out.
    eng_scene = dataclasses.replace(
        scene, fab=dataclasses.replace(scene.fab, min_feature=cfg.engrave_min_feature))

    # The grow below is NOT kerf compensation -- the laser burns where it fires and there is
    # nothing to offset. It corrects a raster-to-vector REGISTRATION bias: marching squares
    # traces the 0.5 level, which runs through the CENTRES of the boundary pixels, so every
    # region comes out inset by half a pixel on every side. The error is proportional to total
    # perimeter, so it is invisible on stage 1's 281 big regions (+0.0037 mean IoU) and
    # savages stage 2's 5402 small ones (+0.0496) -- the same export, the same settings, a
    # 13x difference purely from how much boundary there is. Growing by half a pixel puts the
    # boundary back where the raster said it was.
    pixel = 0.5 * ((scene.panels[0].u_range[1] - scene.panels[0].u_range[0]) / colorid.shape[-1]
                   + (scene.panels[0].v_range[1] - scene.panels[0].v_range[0]) / colorid.shape[-2])
    pieces = decompose.panel_stack_pieces(eng_scene, colorid, names, kerf=pixel)
    n_poly = sum(len(v) for v in pieces.values())
    drawings = build_panel_drawings(scene, {})          # carriers: outline + slots only
    print(f"  {n_poly} polygons over {len(drawings)} sheets")

    # names[0] is 'clear' -- the unengraved sheet, which is the absence of a laser pass and
    # therefore has no layer in the cut file.
    tone_names = list(names[1:])
    paths, counts = export_engrave.export_all(
        drawings, pieces, tone_names, out / "cut", levels=cfg.engrave_levels)
    print(f"  cut files -> {out/'cut'}  ({len(paths)} files)")

    if not a.no_obj:
        export_obj.export_obj(scene, pieces, cfg.thickness, colorid.shape[0],
                              out / "assembly.obj", channel_order=list(tone_names))
        print(f"  assembly -> {out/'assembly.obj'}")

    print("\ngate A: weave feasibility")
    weave = check_weave(scene)
    print(f"  {weave['n_crossings']} crossings in {weave['n_clusters']} clusters, "
          f"max multiplicity {weave['max_multiplicity']}, "
          f"min angle {weave['min_crossing_angle_deg']:.1f} deg, "
          f"widest slot {weave['max_slot_width_mm']:.2f} mm "
          f"(material {weave['material_thickness_mm']:.1f} mm)")
    print(f"  {'PASS' if weave['pass'] else 'FAIL -- ' + str(weave['n_triple_points']) + ' triple points'}")

    print("\ngate B: fabrication round trip (re-render from the cut files)")
    # Intensity is 1 everywhere after re-toning (one laser power per region), so the round
    # trip is fed a matching all-ones intensity rather than stage 1's recorded values.
    inten = (np.ones_like(colorid, dtype=np.float32) if a.retone > 0
             else art["stack_intensity"])
    fab_m, _ = fab_round_trip(scene, table, renderer, targets, names, pieces, inten)
    s_raw, s_fab = res["summary"], fab_m["_summary"]

    # Report the WHOLE bundle across the gate, not just IoU. IoU is a silhouette score, and
    # re-toning does not chase silhouette -- it chases the tone inside it. A gate that reads
    # only IoU would call a round trip that preserved every grey and softened one edge a
    # regression, and would call a round trip that flattened the face to black a pass.
    def _bundle(m):
        vs = [m[w] for w in scene.walls]
        return (float(np.mean([v["ssim"] for v in vs])),
                float(np.mean([v["rmse"] for v in vs])),
                float(np.mean([v["edge_fidelity"] for v in vs])))
    r_ssim, r_rmse, r_edge = _bundle(live)
    f_ssim, f_rmse, f_edge = _bundle(fab_m)
    print(f"  raster solve : mean IoU {s_raw['mean_iou']:.4f}  min {s_raw['min_iou']:.4f}"
          f"  SSIM {r_ssim:.4f}  RMSE {r_rmse:.4f}  edge {r_edge:.4f}")
    print(f"  cut files    : mean IoU {s_fab['mean_iou']:.4f}  min {s_fab['min_iou']:.4f}"
          f"  SSIM {f_ssim:.4f}  RMSE {f_rmse:.4f}  edge {f_edge:.4f}")
    print(f"  fabrication costs {s_raw['mean_iou'] - s_fab['mean_iou']:+.4f} mean IoU, "
          f"{r_ssim - f_ssim:+.4f} SSIM, {f_rmse - r_rmse:+.4f} RMSE")

    print("\ngate C: single-panel ablation")
    abl = ablate_panels(scene, renderer, targets, panel_T)
    print(f"  idle sheets: {abl['n_idle']}/{len(scene.panels)} "
          f"{'-- ' + ', '.join(abl['idle_panels']) if abl['idle_panels'] else '(none)'}")
    by_view = defaultdict(int)
    for r in abl["per_panel"]:
        by_view[r["best_view"]] += 1
    print(f"  sheets by the view they serve most: {dict(by_view)}")

    report = {
        "arm": a.arm,
        "retone_sweeps": a.retone,
        "retone_descent": retone_log,
        "solve_summary": res["summary"],
        "shards_only_summary": res["summary_shards_only"],
        "geometry": res["geometry"],
        "vector": {"n_polygons": n_poly, "regions_per_panel": counts},
        "gate_weave": weave,
        "gate_fab_round_trip": {"views": {w: fab_m[w] for w in scene.walls},
                                "summary": s_fab,
                                "cost_mean_iou": s_raw["mean_iou"] - s_fab["mean_iou"]},
        "gate_ablation": abl,
        "runtime_s": time.perf_counter() - t0,
    }
    (out / "fab_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {out/'fab_report.json'}   ({report['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
