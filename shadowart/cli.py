"""ShadowArt command-line pipeline.

    shadowart demo                         # write demo target images
    shadowart info  --scene scenes/example.yaml
    shadowart run   --scene scenes/example.yaml [--target-a A.png --target-b B.png] --out out/

`run` is the end-to-end path: images + scene -> geometry -> solve (joint 2-wall) ->
preview -> raster2vec -> fabricate -> DXF/SVG. With no targets it uses the demo images.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from .config.io import load_scene
from .geometry.projection import build_projection_table, report_table
from .forward.renderer import Renderer
from .forward import crosstalk as ct
from .solve.initializer import back_project
from .solve.optimizer import solve
from .raster2vec import halftone, features, contours
from .fabricate.joints import build_panel_drawings
from .fabricate import nesting, layers, export_dxf, export_svg
from .targets.image_ops import load_target, make_sample_images, normalize_silhouette
from .preview.wallview import save_wall_comparison
from .preview.render3d import save_scene_3d


def _flatten_polys(geoms):
    from shapely.geometry import MultiPolygon, Polygon
    out = []
    for g in geoms:
        if g.is_empty:
            continue
        if isinstance(g, Polygon):
            out.append(g)
        elif isinstance(g, MultiPolygon):
            out.extend(p for p in g.geoms if not p.is_empty and p.area > 0)
    return out


def raster_to_pieces(scene, opacities):
    """Per panel: opacity -> threshold -> min-feature -> polygons -> kerf offset."""
    fab = scene.fab
    pieces = {}
    total = 0
    for pi, panel in enumerate(scene.panels):
        Hp, Wp = opacities[pi].shape
        su = (panel.u_range[1] - panel.u_range[0]) / Wp
        sv = (panel.v_range[1] - panel.v_range[0]) / Hp
        pixel = 0.5 * (su + sv)
        mask = halftone.threshold(opacities[pi], fab.threshold)
        mask = features.enforce_min_feature(mask, fab.min_feature, pixel)
        polys = contours.mask_to_polygons(mask, panel.u_range, panel.v_range)
        if fab.kerf > 0:
            polys = _flatten_polys([p.buffer(fab.kerf / 2.0, join_style=2) for p in polys])
        pieces[panel.name] = polys
        total += len(polys)
    return pieces, total


def _iou(pred, target, thr=0.5):
    a = pred >= thr; b = target >= thr
    uni = float((a | b).sum())
    return float((a & b).sum()) / uni if uni else 1.0


def _primary_render(scene, table, renderer, opacity, wall_name):
    """Render one wall using ONLY panels currently primary to it (isolates decomposition
    fidelity from opposite-wall cross-talk). `primary_wall_of` replaces the old family
    check -- no panel declares a wall, it's computed from projected geometry."""
    from .geometry.projection import primary_wall_of
    opf = opacity.copy()
    for i, p in enumerate(scene.panels):
        if primary_wall_of(scene, table, p) != wall_name:
            opf[i] = 0.0
    return renderer.render_np(opf)[wall_name]


def _print_fragment_stats(scene, table, renderer, fragments, resolved, opacity, targets):
    import numpy as _np
    from .solve.decompose import count_collisions
    print("\n=== fragment statistics (shard sizes as equivalent side length) ===")
    safe_mm = scene.fab.min_feature * 1000 * 1.5           # a comfortable cut size
    for wn, wall in (("A", "Wall A"), ("B", "Wall B")):
        sides = _np.sqrt(_np.array([f["phys_mm2"] for f in fragments if f["wall"] == wn]))
        if sides.size == 0:
            continue
        pct = _np.percentile(sides, [0, 10, 50, 90, 100])
        print(f"{wall}: {sides.size} shards   side(mm) "
              f"min {pct[0]:.0f} | p10 {pct[1]:.0f} | median {pct[2]:.0f} | "
              f"p90 {pct[3]:.0f} | max {pct[4]:.0f}")
        per_panel = {}
        for f in fragments:
            if f["wall"] == wn:
                per_panel[f["panel"]] = per_panel.get(f["panel"], 0) + 1
        print("        per panel: " + ", ".join(f"{k}:{v}" for k, v in sorted(per_panel.items())))
        hist, edges = _np.histogram(sides, bins=6)
        for i, c in enumerate(hist):
            bar = "#" * int(40 * c / max(hist.max(), 1))
            print(f"        {edges[i]:5.0f}-{edges[i+1]:5.0f} mm | {bar} {c}")
        below = int((sides < safe_mm).sum())
        if below:
            print(f"        NOTE: {below} shards below ~{safe_mm:.0f} mm (near the cut limit)")
    for wn, wall in (("A", "Wall A"), ("B", "Wall B")):
        iou = _iou(_primary_render(scene, table, renderer, opacity, wn), targets[wn])
        print(f"{wall}: reconstruction IoU {iou:.3f} (shards-only, excl. cross-talk)")
    print(f"collisions: {resolved} resolved, {count_collisions(scene, opacity)} remaining")


def cmd_demo(args):
    paths = make_sample_images(args.out or "examples/targets")
    print("wrote demo targets:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


def cmd_info(args):
    scene = load_scene(args.scene)
    table = build_projection_table(scene)
    print(report_table(scene, table))
    out = Path(args.out or "out"); out.mkdir(parents=True, exist_ok=True)
    p = save_scene_3d(scene, out / "scene_3d.png")
    print(f"\n3D preview: {p}")


def _run_color(scene, out, args):
    """CMYK / coloured-perspex pipeline: per-shard palette colour + colour projection."""
    from collections import Counter
    from .targets import color as C
    from .solve import decompose
    from .fabricate import export_color
    from .fabricate.export_ply import export_ply
    from .preview.wallview import save_color_comparison
    from .preview.interactive3d import build_interactive, _palette_color_of

    if not (args.target_a and args.target_b):
        raise SystemExit("colour mode needs --target-a and --target-b "
                         "(e.g. examples/cmyk1.jpeg examples/cmyk2.png)")
    names = C.palette_names(scene.color_palette)
    wr = scene.solve.wall_res
    targets = {"A": C.load_color_target(args.target_a, wr, white_thr=scene.white_threshold),
               "B": C.load_color_target(args.target_b, wr, white_thr=scene.white_threshold)}
    print(f"colour targets: A={args.target_a}  B={args.target_b}")
    print(f"perspex palette: {[n for n in names if n != 'clear']}")

    table = build_projection_table(scene)
    save_scene_3d(scene, out / "scene_3d.png")
    renderer = Renderer(scene, table)

    print("splitting into CMYK channels + building stacked shards ...")
    opacity, fragments, colorid, resolved, stack_depths = decompose.fragment_shards_cmyk(
        scene, table, targets, names=names, white_thr=scene.white_threshold,
        max_layers=scene.color_max_layers)
    panel_T = C.transmit_lut(names)[colorid]                # [P,Hp,Wp,3]
    pred_rgb = renderer.render_color_np(panel_T)

    from . import metrics
    print(metrics.format_accuracy_report(metrics.evaluate_wall_accuracy(targets, pred_rgb)))

    save_color_comparison(targets, pred_rgb, out / "preview_final.png")
    for fam in scene.walls:
        C.save_cmyk_channels(targets[fam], out / "cmyk_channels", fam)
    np.save(out / "opacity.npy", opacity)
    np.save(out / "colorid.npy", colorid)

    print("\n=== CMYK shard counts (per channel) ===")
    for fam in scene.walls:
        cnt = Counter(f["channel"] for f in fragments if f["wall"] == fam)
        print(f"Wall {fam}: {sum(cnt.values())} shards  "
              + ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items())))
    if stack_depths:
        sd = np.asarray(stack_depths)
        hist = ", ".join(f"{d}-layer:{int((sd == d).sum())}" for d in range(1, int(sd.max()) + 1))
        print(f"stack depth per shard region (tone/mixing): {hist}")
    print(f"collisions: {resolved} resolved, {decompose.count_collisions(scene, opacity)} remaining")

    print("vectorising per-channel + exporting per-colour cut files ...")
    pieces = decompose.panel_channel_pieces(scene, colorid, names)
    total = sum(len(v) for v in pieces.values())

    build_interactive(scene, table, opacity, None, out / "scene_interactive.html",
                      rays=40, auto_open=False, panel_colorid=colorid, names=names,
                      wall_rgb=pred_rgb, pieces=pieces)

    structure = build_panel_drawings(scene, {})             # clear carriers: outline + slots
    if "dxf" in scene.fab.formats:
        export_dxf.export_all_dxf(structure, out / "cut" / "structure")
    if "svg" in scene.fab.formats:
        export_svg.export_all_svg(structure, out / "cut" / "structure")
    _, counts = export_color.export_all_color(scene, pieces, colorid, names,
                                              out / "cut", formats=scene.fab.formats)
    export_ply(scene, pieces, scene.material_thickness,
               _palette_color_of(scene, colorid, names), out / "shards.ply")
    print(f"  {total} shards; per-colour cut files: "
          + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    print(f"wrote {out/'cut'} (per-colour + structure), shards.ply, scene_interactive.html, "
          "preview_final.png, cmyk_channels/")


def _run_color_overlap(scene, out, args):
    """'Stochastic Shard Overlap': C/M/Y/K shards interleave onto the SAME depth plane(s)
    (typically 1-2 per family) instead of each channel getting its own plane. Secondary and
    tertiary colours come from laminating shards at the same (u,v) footprint but a slightly
    different micro depth, so subtractive mixing happens right at that spot. Placement is
    randomised (which plane, plus a small per-channel jitter capped by that panel's physical
    penumbra) so the structure reads as a chaotic cloud of overlapping brushstrokes rather
    than a tidy per-channel tiling. Ships as one dense .obj (plus per-slot cut files, since
    laminated sub-sheets still need cutting)."""
    from collections import Counter
    from .targets import color as C
    from .solve import decompose
    from .fabricate import export_color, export_obj as export_obj_mod
    from .fabricate.export_ply import export_ply_stack
    from .preview.wallview import save_color_comparison
    from .preview.interactive3d import build_interactive

    if not (args.target_a and args.target_b):
        raise SystemExit("colour mode needs --target-a and --target-b "
                         "(e.g. examples/cmyk1.jpeg examples/cmyk2.png)")
    names = C.palette_names(scene.color_palette)
    channel_order = [n for n in names if n != "clear"]
    wr = scene.solve.wall_res
    targets = {"A": C.load_color_target(args.target_a, wr, white_thr=scene.white_threshold),
               "B": C.load_color_target(args.target_b, wr, white_thr=scene.white_threshold)}
    print(f"colour targets: A={args.target_a}  B={args.target_b}  (stochastic shard overlap)")
    print(f"perspex palette: {channel_order}   max stack: {scene.color_max_stack}")
    print(f"fabrication: kerf {scene.fab.kerf * 1000:.2f} mm (applied to every shard polygon "
          f"in panel_stack_pieces -- shared by the cut files and shards.obj/.ply)")

    table = build_projection_table(scene)
    save_scene_3d(scene, out / "scene_3d.png")
    renderer = Renderer(scene, table)

    from .solve import search
    sp = scene.solve
    weights = {"ssim": sp.score_ssim_weight, "edge": sp.score_edge_weight,
               "crosstalk": sp.score_crosstalk_weight, "joint_thresh": 0.3}
    dmg = sp.damage_weight
    multi = sp.restarts > 1 or sp.panel_restarts > 1 or sp.time_budget > 0 or sp.search_panels
    if dmg == 0.0 and multi:
        dmg = 0.5                          # actually use the greedy host-selection when searching
    cred = sp.credit_weight or None

    print("interleaving C/M/Y/K shards onto shared depth planes (stochastic overlap) ...")
    if sp.search_panels:
        best = search.multi_run_panels(
            scene, targets, names,
            panel_restarts=sp.panel_restarts, shard_restarts=sp.restarts,
            panel_time_budget=(sp.time_budget or None), shard_time_budget=(sp.time_budget or None),
            damage_weight=dmg, credit_weight=cred, weights=weights)
        scene = dataclasses.replace(scene, panels=best["panels"])   # export code below uses chosen panels
        table, renderer = best["table"], best["renderer"]
        sb = best["shard_best"]
        print(f"panel search: kept layout seed={best['panel_seed']} of {best['n_layouts']} "
              f"(composite score {best['score']:.4f})")
    else:
        sb = search.multi_run_shards(
            scene, table, targets, names, renderer,
            restarts=sp.restarts, time_budget=(sp.time_budget or None),
            damage_weight=dmg, credit_weight=cred, weights=weights)
    print(f"shard search: kept seed={sb['seed']} of {sb['n_runs']} runs "
          f"(composite score {sb['score']:.4f})")

    stack_colorid, opacity, fragments = sb["stack_colorid"], sb["opacity"], sb["fragments"]
    resolved, stack_depths = sb["resolved"], sb["stack_depths"]
    budget_stats, stack_intensity, pred_rgb = sb["budget_stats"], sb["stack_intensity"], sb["pred_rgb"]
    S = stack_colorid.shape[0]

    from . import metrics
    print(metrics.format_accuracy_report(metrics.evaluate_wall_accuracy(targets, pred_rgb)))

    save_color_comparison(targets, pred_rgb, out / "preview_final.png")
    for fam in scene.walls:
        C.save_cmyk_channels(targets[fam], out / "cmyk_channels", fam)
    np.save(out / "opacity.npy", opacity)
    np.save(out / "stack_colorid.npy", stack_colorid)

    print("\n=== shard regions vs. fabrication ceiling ===")
    for fam in scene.walls:
        b = budget_stats.get(fam, {})
        ceil = b.get("target")
        ceil_s = f"{ceil}" if ceil else "unlimited"
        mult = b.get("spacing_multiplier", 1.0)
        note = " (coarsened to fit)" if mult != 1.0 else " (natural, detail-biased count)"
        print(f"Wall {fam}: {b.get('achieved', '?')} shard regions "
              f"(ceiling {ceil_s}{note})  |  penumbra-aware min feature: "
              f"{b.get('penumbra_min_feature_mm', 0.0):.2f} mm")

    print("\n=== overlap shard counts (per channel, across all stack slots) ===")
    for fam in scene.walls:
        cnt = Counter(f["channel"] for f in fragments if f["wall"] == fam)
        print(f"Wall {fam}: {sum(cnt.values())} laminated layers  "
              + ", ".join(f"{k}:{v}" for k, v in sorted(cnt.items())))
    if stack_depths:
        sd = np.asarray(stack_depths)
        hist = ", ".join(f"{d}-stack:{int((sd == d).sum())}" for d in range(1, int(sd.max()) + 1))
        print(f"stack depth per shard region (secondary/tertiary mixing): {hist}")
    if fragments:
        jpx = np.asarray([f["jitter_px"] for f in fragments])
        sig = np.asarray([f["softening_sigmas"] for f in fragments])
        print(f"scatter: jitter_px min {jpx.min()} | median {int(np.median(jpx))} | max {jpx.max()}")
        print(f"  softening vs. physical penumbra: {np.median(sig):.1f}x sigma (median), "
              f"{sig.max():.1f}x sigma (max) -- >1x means the cast shadow is being "
              f"deliberately softened by this much to buy extra scrambling")
    print(f"collisions: {resolved} resolved, {decompose.count_collisions(scene, opacity)} remaining")

    print("vectorising stack slots + exporting per-colour cut files + dense .obj ...")
    stack_pieces = decompose.panel_stack_pieces(scene, stack_colorid, names)
    total = sum(len(v) for v in stack_pieces.values())

    poly_channel = {id(poly): ch for items in stack_pieces.values() for poly, ch, _s in items}
    flat_pieces = {name: [poly for poly, _ch, _s in items] for name, items in stack_pieces.items()}
    stack_color_of = lambda panel, poly: tuple(C.display_rgb(poly_channel.get(id(poly), "clear")))

    build_interactive(scene, table, opacity, None, out / "scene_interactive.html",
                      rays=40, auto_open=False, wall_rgb=pred_rgb, pieces=flat_pieces,
                      color_of=stack_color_of)

    structure = build_panel_drawings(scene, {})             # clear carriers: outline + slots
    if "dxf" in scene.fab.formats:
        export_dxf.export_all_dxf(structure, out / "cut" / "structure")
    if "svg" in scene.fab.formats:
        export_svg.export_all_svg(structure, out / "cut" / "structure")

    pieces_by_slot = [{p.name: [] for p in scene.panels} for _ in range(S)]
    for panel_name, items in stack_pieces.items():
        for poly, ch, slot in items:
            pieces_by_slot[slot][panel_name].append(poly)
    counts = Counter()
    for s in range(S):
        _, cnt = export_color.export_all_color(scene, pieces_by_slot[s], stack_colorid[s], names,
                                               out / "cut" / f"stack{s}", formats=scene.fab.formats)
        counts.update(cnt)

    export_ply_stack(scene, stack_pieces, scene.material_thickness, S,
                     C.transmit_rgb, channel_order, out / "shards.ply")
    export_obj_mod.export_obj(scene, stack_pieces, scene.material_thickness, S,
                              out / "shards.obj", channel_order=channel_order)
    print(f"  {total} laminated shards; per-colour cut files: "
          + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    print(f"wrote {out/'shards.obj'} (shards + wall/panel/light rig groups for orientation), "
          f"shards.ply, {out/'cut'} (per-slot per-colour + structure), scene_interactive.html, "
          "preview_final.png, cmyk_channels/")


def cmd_run(args):
    scene = load_scene(args.scene)
    out = Path(args.out or "out"); out.mkdir(parents=True, exist_ok=True)

    # CLI overrides for the multi-run search knobs (only for keys the user actually passed,
    # so an unset flag leaves the scene YAML value in place). Guarded with getattr so the
    # shared mono path -- which reuses this function -- is untouched.
    ov = {}
    if getattr(args, "restarts", None) is not None:       ov["restarts"] = args.restarts
    if getattr(args, "panel_restarts", None) is not None: ov["panel_restarts"] = args.panel_restarts
    if getattr(args, "time_budget", None) is not None:    ov["time_budget"] = args.time_budget
    if getattr(args, "search_panels", False):             ov["search_panels"] = True
    if getattr(args, "damage_weight", None) is not None:  ov["damage_weight"] = args.damage_weight
    if getattr(args, "diagonal_frac", None) is not None:  ov["diagonal_frac"] = args.diagonal_frac
    if ov:
        scene = dataclasses.replace(scene, solve=dataclasses.replace(scene.solve, **ov))
    if getattr(args, "palette", None):
        from .targets import color as _C
        scene = dataclasses.replace(scene, color_palette=list(_C.PALETTES[args.palette]))

    # Isolate a centered subject on a plain background before solving -- the pipeline
    # reconstructs a clear silhouette far better than a full-frame image (see
    # image_ops.remove_background). Writes <out>/nobg_{a,b}.png and points the run at them.
    if getattr(args, "remove_bg", False) and args.target_a and args.target_b:
        from .targets.image_ops import remove_background
        for slot in ("target_a", "target_b"):
            src = getattr(args, slot)
            dst = out / f"nobg_{slot[-1]}.png"
            _, cov, ok = remove_background(src, str(dst))
            if ok:
                print(f"removed background: {src} -> {dst}  (subject {cov*100:.0f}% of frame)")
                setattr(args, slot, str(dst))
            else:
                print(f"WARNING: background removal untrustworthy on {src} "
                      f"(subject {cov*100:.0f}% of frame -- background is dark/busy or the "
                      f"subject blends in). Keeping the ORIGINAL image; supply a manual "
                      f"cut-out (subject on white) for a clean result.")

    if getattr(args, "color", False):
        if getattr(args, "color_mode", "weave") == "overlap":
            return _run_color_overlap(scene, out, args)
        return _run_color(scene, out, args)

    # 1. targets ---------------------------------------------------------
    if args.target_a and args.target_b:
        ta, tb = args.target_a, args.target_b
    else:
        print("no targets given -> generating demo images")
        d = make_sample_images(out / "targets")
        ta, tb = d["a_heart"], d["b_star"]
    targets = {"A": load_target(ta, scene.solve.wall_res),
               "B": load_target(tb, scene.solve.wall_res)}
    print(f"targets: A={ta}  B={tb}")

    # 2. geometry --------------------------------------------------------
    table = build_projection_table(scene)
    print(report_table(scene, table))
    save_scene_3d(scene, out / "scene_3d.png")
    renderer = Renderer(scene, table)

    # 3. decompose / solve ----------------------------------------------
    fragments = resolved = None
    if scene.solve.mode == "partition":
        from .solve import decompose
        print("fragmenting image into shards across depth planes ...")
        opacity, fragments, resolved = decompose.fragment_shards(scene, table, targets)
    else:
        print("back-projection init ...")
        init = back_project(scene, table, targets)
        save_wall_comparison(targets, renderer.render_np(init), out / "preview_init.png")
        print(f"optimising {scene.solve.iters} iters ...")
        opacity, _ = solve(scene, renderer, targets, init_opacity=init)
    pred = renderer.render_np(opacity)
    cross = {w: ct.crosstalk_only(renderer, opacity, w).detach().cpu().numpy()
             for w in scene.walls}
    save_wall_comparison(targets, pred, out / "preview_final.png", crosstalk=cross)
    np.save(out / "opacity.npy", opacity)
    for w in scene.walls:
        mse = float(((pred[w] - targets[w]) ** 2).mean())
        print(f"  wall {w}: final MSE {mse:.5f}   mean cross-talk {cross[w].mean():.4f}")
    if fragments is not None:
        _print_fragment_stats(scene, table, renderer, fragments, resolved, opacity, targets)

    from .preview.interactive3d import build_interactive
    html, _ = build_interactive(scene, table, opacity, pred,
                                out / "scene_interactive.html", rays=40, auto_open=False)
    print(f"interactive 3D preview: {html}  (open it, or run `shadowart view`)")

    # 4. raster -> vector ------------------------------------------------
    print("vectorising (threshold -> min-feature -> contours -> kerf) ...")
    pieces, total = raster_to_pieces(scene, opacity)
    print(f"  {total} cut pieces across {len(scene.panels)} panels")

    # 5. fabricate -------------------------------------------------------
    drawings = build_panel_drawings(scene, pieces)
    placements = nesting.nest(drawings, scene.fab.sheet_size)
    oversize = [p.name for p in placements if p.oversize]
    if oversize:
        print(f"  WARNING: panels exceed stock sheet {scene.fab.sheet_size} m: {oversize} "
              "(tile or use a larger sheet)")
    grouped = layers.group_mono(drawings)          # {'mono': drawings}
    written = []
    for layer_name, dlist in grouped.items():
        cut_dir = out / "cut" / layer_name
        if "dxf" in scene.fab.formats:
            written += export_dxf.export_all_dxf(dlist, cut_dir)
        if "svg" in scene.fab.formats:
            written += export_svg.export_all_svg(dlist, cut_dir)
    print(f"wrote {len(written)} cut files under {out / 'cut'}")
    print("\ndone. previews: preview_init.png, preview_final.png, scene_3d.png")


def cmd_normalize(args):
    """Normalise one or more silhouettes to the same square canvas + scale."""
    out_dir = Path(args.out_dir or "examples"); out_dir.mkdir(parents=True, exist_ok=True)
    for src in args.inputs:
        dst = out_dir / f"{Path(src).stem}_norm.png"
        _, size = normalize_silhouette(src, dst, size=args.size,
                                       content_frac=args.content, align=args.align)
        print(f"  {src}  ->  {dst}  ({size[0]}x{size[1]})")


def cmd_segment(args):
    """Isolate a centered subject on a plain background -> subject on white."""
    from .targets.image_ops import remove_background
    out_dir = Path(args.out_dir or "examples"); out_dir.mkdir(parents=True, exist_ok=True)
    for src in args.inputs:
        dst = out_dir / f"{Path(src).stem}_nobg.png"
        _, cov, ok = remove_background(src, str(dst), bg_tol=args.bg_tol, sat_tol=args.sat_tol)
        flag = "" if ok else "  [UNTRUSTWORTHY -- kept original; needs a manual cut-out]"
        print(f"  {src}  ->  {dst}  (subject {cov*100:.0f}% of frame){flag}")


def cmd_view(args):
    """Open the interactive 3D window from a previously solved opacity map."""
    scene = load_scene(args.scene)
    out = Path(args.out or "out")
    oppath = Path(args.opacity) if args.opacity else out / "opacity.npy"
    if not oppath.exists():
        raise SystemExit(f"no solved opacity at {oppath} — run `shadowart run` first "
                         "(or pass --opacity).")
    opacity = np.load(oppath)
    table = build_projection_table(scene)
    renderer = Renderer(scene, table)
    pred = renderer.render_np(opacity)
    from .preview.interactive3d import build_interactive
    html, _ = build_interactive(scene, table, opacity, pred,
                                out / "scene_interactive.html",
                                rays=args.rays, auto_open=not args.no_open,
                                show_panels=args.show_panels, shard_thickness=args.thickness)
    print(f"interactive 3D preview: {html}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="shadowart", description="Computational shadow-art pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("demo", help="write demo target images")
    pd.add_argument("--out")
    pd.set_defaults(func=cmd_demo)

    pi = sub.add_parser("info", help="print projection table + 3D preview")
    pi.add_argument("--scene", required=True)
    pi.add_argument("--out")
    pi.set_defaults(func=cmd_info)

    pr = sub.add_parser("run", help="end-to-end solve + preview + export")
    pr.add_argument("--scene", required=True)
    pr.add_argument("--target-a")
    pr.add_argument("--target-b")
    pr.add_argument("--color", action="store_true",
                    help="CMYK coloured-perspex mode (each shard a stock perspex colour)")
    pr.add_argument("--color-mode", choices=["weave", "overlap"], default="weave",
                    help="weave = each channel scattered onto its own depth plane (legacy "
                         "layer-separation); overlap = Stochastic Shard Overlap: channels "
                         "laminated on the SAME 1-2 planes for direct subtractive mixing, "
                         "output as a single dense .obj (default: weave)")
    pr.add_argument("--palette", choices=["cmyk", "muted", "noir"], default=None,
                    help="perspex palette preset (overrides scene color.palette/preset): "
                         "cmyk (default), muted (dusty low-chroma), noir (tonal grey)")
    pr.add_argument("--remove-bg", action="store_true",
                    help="isolate a centered subject on a plain background before solving "
                         "(the pipeline reconstructs a clear silhouette far better than a "
                         "full-frame image); writes <out>/nobg_a.png, nobg_b.png")
    # --- multi-run search (overlap colour mode); all default None so scene YAML wins if unset ---
    pr.add_argument("--restarts", type=int, default=None,
                    help="restart shard placement N times, keep best by composite score")
    pr.add_argument("--panel-restarts", type=int, default=None,
                    help="with --search-panels, build N candidate panel layouts, keep best")
    pr.add_argument("--time-budget", type=float, default=None,
                    help="wallclock sec cap; stop at whichever of --restarts / this is hit first")
    pr.add_argument("--search-panels", action="store_true",
                    help="greedy panel-layout search instead of the scene's fixed panels")
    pr.add_argument("--damage-weight", type=float, default=None,
                    help=">0 = cross-talk-damage-minimising host selection "
                         "(auto 0.5 when multi-running)")
    pr.add_argument("--diagonal-frac", type=float, default=None,
                    help="force this SHARE (0..1) of shards onto diagonal planes so the "
                         "piece isn't trivial (the greedy otherwise leaves diagonals empty); "
                         "requires diagonal panels in the scene")
    pr.add_argument("--out")
    pr.set_defaults(func=cmd_run)

    pn = sub.add_parser("normalize", help="resize silhouettes to the same square canvas + scale")
    pn.add_argument("--in", dest="inputs", nargs="+", required=True, help="input image(s)")
    pn.add_argument("--out-dir", help="output dir (default: examples)")
    pn.add_argument("--size", type=int, default=800, help="output canvas size in px (default 800)")
    pn.add_argument("--content", type=float, default=0.9, help="piece height as fraction of canvas")
    pn.add_argument("--align", default="bottom", choices=["bottom", "center", "top"])
    pn.set_defaults(func=cmd_normalize)

    ps = sub.add_parser("segment", help="isolate a centered subject on a plain background")
    ps.add_argument("--in", dest="inputs", nargs="+", required=True, help="input image(s)")
    ps.add_argument("--out-dir", help="output dir (default: examples)")
    ps.add_argument("--bg-tol", type=float, default=0.13,
                    help="colour distance from the corner background to treat as background")
    ps.add_argument("--sat-tol", type=float, default=0.12,
                    help="max saturation for a pixel to count as background")
    ps.set_defaults(func=cmd_segment)

    pv = sub.add_parser("view", help="open the interactive 3D window (after a run)")
    pv.add_argument("--scene", required=True)
    pv.add_argument("--opacity", help="path to opacity.npy (default: <out>/opacity.npy)")
    pv.add_argument("--rays", type=int, default=40, help="number of light rays to draw (0=none)")
    pv.add_argument("--thickness", type=float, default=0.02,
                    help="visual shard thickness in metres (default 0.02, exaggerated)")
    pv.add_argument("--show-panels", action="store_true", help="also draw the faint panel planes")
    pv.add_argument("--no-open", action="store_true", help="write HTML but don't open a browser")
    pv.add_argument("--out")
    pv.set_defaults(func=cmd_view)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
