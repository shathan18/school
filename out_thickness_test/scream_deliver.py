"""FINAL fabricable Scream deliverables, for BOTH colour and 2-tone (~300 chunky shards each).
The screaming figure gets its own dense region; background left coarse. For each option writes:
  preview_final.png (source vs reconstruction, both walls), on_wall.png (installation in the room
  corner), scene.html (interactive rotatable 3D with the shards), scene_front/back.png, and the
  DXF/SVG per-colour cut files + shards.obj under cut/.

  py out_thickness_test/scream_deliver.py
"""
import sys, os, dataclasses, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import hsv_to_rgb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from PIL import Image
from scipy import ndimage
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.solve import decompose
from shadowart.targets import color as C
from shadowart.metrics import _luma
from shadowart.preview.render3d import save_scene_3d
from shadowart.preview.interactive3d import build_interactive
from shadowart.preview.wallview import save_color_comparison
from shadowart.fabricate.joints import build_panel_drawings
from shadowart.fabricate import export_color, export_obj as export_obj_mod, export_dxf, export_svg
import objectseg as OS

src = open("out_thickness_test/scream_run.py").read()
seg = src[src.index("def load_scene_and_figure"):src.index("# ---- targets")]
ns = {"np": np, "ndimage": ndimage, "C": C, "Image": Image}; exec(seg, ns)

scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res
scene0 = dataclasses.replace(scene0, overlap_shard_budget=300)
names = C.palette_names(scene0.color_palette); channel_order = [n for n in names if n != "clear"]
scene_raw, fig_raw = ns["load_scene_and_figure"]("examples/scream_src.jpg", "examples/scream_figure_nobg.png", scene0.white_threshold)
a_col, fig_mask = ns["fit_flip"](scene_raw, fig_raw, WR, scene0.white_threshold)
B = C.load_color_target("examples/munch_self_nobg.png", WR, white_thr=scene0.white_threshold)
subj = C.subject_mask(a_col, scene0.white_threshold)
lum = _luma(a_col)
a_2tone = np.where((lum < 0.5)[..., None], np.float32(0.10), np.float32(0.75)).repeat(3, 2)
a_2tone[~subj] = 1.0
TARGETS = {"colour": a_col, "2tone": a_2tone}

sp = scene0.solve
labA = OS.segment_objects(a_col, subj, k=12, min_frac=0.004, max_objects=40)
fl = int(labA.max()) + 1; labA[fig_mask & subj] = fl
labB = OS.segment_objects(B, C.subject_mask(B, scene0.white_threshold), k=12, min_frac=0.004, max_objects=40)
rmasks = {"A": labA, "B": labB}; rscales = {"A": {fl: 0.4}}
panels, _ = build_panels_greedy(scene0, count=14, mode="deliberate", K=16, targets={"A": a_col, "B": B},
                                seed=2, angle_deg_range=(5, 85), anchor_range=sp.search_anchor_range,
                                standoff=sp.search_standoff, mag_cap=sp.search_mag_cap,
                                u_size_range=sp.search_u_size_range, v_range=sp.search_v_range)
ts = dataclasses.replace(scene0, panels=panels, solve=dataclasses.replace(sp, diagonal_frac=0.0))
table = build_projection_table(ts); R = Renderer(ts, table)


def wall_corners(w):
    o = w.origin
    return np.array([o, o + w.axis_u * w.width, o + w.axis_u * w.width + w.axis_v * w.height, o + w.axis_v * w.height])


def installation(pred, path):
    fig = plt.figure(figsize=(8, 7.5)); ax = fig.add_subplot(111, projection="3d")
    for wn in ("A", "B"):
        w = ts.walls[wn]; gs = 120
        small = np.asarray(Image.fromarray((np.clip(pred[wn], 0, 1) * 255).astype(np.uint8)).resize((gs, gs), Image.LANCZOS)) / 255.0
        u = np.linspace(0, w.width, gs); v = np.linspace(0, w.height, gs); U, V = np.meshgrid(u, v)
        if w.plane == "x":
            X = np.full_like(U, w.offset); Y = U; Z = w.z0 + V
        else:
            X = U; Y = np.full_like(U, w.offset); Z = w.z0 + V
        ax.plot_surface(X, Y, Z, facecolors=np.dstack([small, np.ones((gs, gs))]), rstride=1, cstride=1, shade=False, antialiased=False, linewidth=0)
    for p in ts.panels:
        (u0, u1), (v0, v1) = p.u_range, p.v_range
        q = p.uv_to_xyz(np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]))
        ax.add_collection3d(Poly3DCollection([q], facecolors=[(*hsv_to_rgb(((p.angle % np.pi) / np.pi, 0.6, 0.85)), 0.5)], edgecolors="k", linewidths=0.2))
    pts = np.concatenate([wall_corners(ts.walls["A"]), wall_corners(ts.walls["B"])]
                         + [p.uv_to_xyz(np.array([[p.u_range[0], p.v_range[0]], [p.u_range[1], p.v_range[1]]])) for p in ts.panels])
    lo, hi = pts.min(0), pts.max(0); span = (hi - lo).max(); mid = (hi + lo) / 2
    for setl, m in ((ax.set_xlim, mid[0]), (ax.set_ylim, mid[1]), (ax.set_zlim, mid[2])):
        setl(m - span / 2, m + span / 2)
    ax.view_init(elev=20, azim=45); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    plt.tight_layout(); plt.savefig(path, dpi=115, bbox_inches="tight"); plt.close()


for opt, aT in TARGETS.items():
    OUT = Path(f"out_scream_{opt}"); OUT.mkdir(exist_ok=True)
    tgts = {"A": aT, "B": B}
    sc, op, fr, rs, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, tgts, names=names, white_thr=ts.white_threshold, max_stack=ts.color_max_stack,
        seed=2, damage_weight=0.5, credit_weight=0.5, match_tol=0.30, colour_blend=0.6,
        region_masks=rmasks, region_scales=rscales)
    S = sc.shape[0]
    pred = R.render_color_np(C.stack_transmit_lut(names, sc, si))
    nsh = int(bs.get("A", {}).get("achieved", 0)) + int(bs.get("B", {}).get("achieved", 0))
    print(f"[{opt}] {nsh} shard regions")

    # source (real colour Scream) vs reconstruction, both walls
    save_color_comparison({"A": a_col, "B": B}, pred, OUT / "preview_final.png")
    installation(pred, OUT / "on_wall.png")
    save_scene_3d(ts, str(OUT / "scene_front.png"), elev=18, azim=-60, title=f"{opt} - front")
    save_scene_3d(ts, str(OUT / "scene_back.png"), elev=18, azim=120, title=f"{opt} - from behind")

    stack_pieces = decompose.panel_stack_pieces(ts, sc, names)
    poly_channel = {id(poly): ch for items in stack_pieces.values() for poly, ch, _s in items}
    flat = {n: [poly for poly, _ch, _s in items] for n, items in stack_pieces.items()}
    color_of = lambda panel, poly: tuple(C.display_rgb(poly_channel.get(id(poly), "clear")))
    try:
        build_interactive(ts, table, op, None, str(OUT / "scene.html"), rays=40, auto_open=False,
                          wall_rgb=pred, pieces=flat, color_of=color_of)
        print(f"[{opt}] wrote scene.html")
    except Exception as e:
        print(f"[{opt}] scene.html skipped: {e}")

    structure = build_panel_drawings(ts, {})
    export_dxf.export_all_dxf(structure, OUT / "cut" / "structure")
    export_svg.export_all_svg(structure, OUT / "cut" / "structure")
    pieces_by_slot = [{p.name: [] for p in ts.panels} for _ in range(S)]
    for pn, items in stack_pieces.items():
        for poly, ch, slot in items:
            pieces_by_slot[slot][pn].append(poly)
    total = 0
    for s in range(S):
        _, cnt = export_color.export_all_color(ts, pieces_by_slot[s], sc[s], names, OUT / "cut" / f"stack{s}", formats=ts.fab.formats)
        total += sum(cnt.values())
    try:
        export_obj_mod.export_obj(ts, stack_pieces, ts.material_thickness, S, str(OUT / "shards.obj"), channel_order=channel_order)
    except Exception as e:
        print(f"[{opt}] obj skipped: {e}")
    print(f"[{opt}] wrote {total} cut polys -> {OUT}/  (preview_final, on_wall, scene.html, scene_front/back, cut/, shards.obj)")

print("done: out_scream_colour/ and out_scream_2tone/")
