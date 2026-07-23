"""Finish the Scream deliverable from the saved optimizer opacity (no re-solve): render the honest
FABRICABLE version via dithering (tone -> cuttable dots, since partial opacity isn't buildable),
rewrite walls.png, write scene.html, and export the DXF/SVG cut files (reporting the min-feature
loss). Rebuilds the same seed-2 panel layout the opacity was solved on."""
import sys, os, dataclasses, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from PIL import Image
from scipy import ndimage
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart.metrics import _luma
from shadowart.raster2vec import halftone
from shadowart.preview.render3d import save_scene_3d
from shadowart.preview.interactive3d import build_interactive
from shadowart.cli import raster_to_pieces
from shadowart.fabricate.joints import build_panel_drawings
from shadowart.fabricate import nesting, layers, export_dxf, export_svg


def load_scene_and_figure(scene_path, fig_path, white_thr):
    s = Image.open(scene_path).convert("RGB"); n = Image.open(fig_path).convert("RGB")
    if n.size != s.size:
        n = n.resize(s.size, Image.LANCZOS)
    return np.asarray(s, np.float32) / 255.0, C.subject_mask(np.asarray(n, np.float32) / 255.0, white_thr)


def fit_flip(rgb, mask, wall_res, white_thr, content_frac=0.92):
    Hn, Wn = wall_res
    sub = ndimage.binary_opening(C.subject_mask(rgb, white_thr), iterations=2)
    lbl, n = ndimage.label(sub)
    if n:
        cnt = np.bincount(lbl.ravel()); cnt[0] = 0
        ys, xs = np.where(lbl == int(cnt.argmax()))
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    else:
        y0, y1, x0, x1 = 0, rgb.shape[0], 0, rgb.shape[1]
    rc, mc = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    ch, cw = rc.shape[:2]
    scl = content_frac * min(Wn / cw, Hn / ch)
    nw, nh = max(1, int(round(cw * scl))), max(1, int(round(ch * scl)))
    ri = Image.fromarray((np.clip(rc, 0, 1) * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    mi = Image.fromarray((mc.astype(np.uint8) * 255)).resize((nw, nh), Image.NEAREST)
    Rc = np.ones((Hn, Wn, 3), np.float32); M = np.zeros((Hn, Wn), bool)
    oy, ox = (Hn - nh) // 2, (Wn - nw) // 2
    Rc[oy:oy + nh, ox:ox + nw] = np.asarray(ri, np.float32) / 255.0
    M[oy:oy + nh, ox:ox + nw] = np.asarray(mi) > 127
    return np.flipud(Rc).copy(), np.flipud(M).copy()


OUT = Path("out_scream_final"); SEED = 2
scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res
scene_raw, fig_raw = load_scene_and_figure("examples/scream_src.jpg", "examples/scream_figure_nobg.png", scene0.white_threshold)
a_scene, _ = fit_flip(scene_raw, fig_raw, WR, scene0.white_threshold)
B = C.load_color_target("examples/munch_self_nobg.png", WR, white_thr=scene0.white_threshold)
tg = {"A": (1.0 - _luma(a_scene)).astype(np.float32), "B": (1.0 - _luma(B)).astype(np.float32)}
sp = scene0.solve
panels, _ = build_panels_greedy(scene0, count=14, mode="deliberate", K=16, targets={"A": a_scene, "B": B},
                                seed=SEED, angle_deg_range=(5, 85), anchor_range=sp.search_anchor_range,
                                standoff=sp.search_standoff, mag_cap=sp.search_mag_cap,
                                u_size_range=sp.search_u_size_range, v_range=sp.search_v_range)
ts = dataclasses.replace(scene0, panels=panels, solve=dataclasses.replace(sp, diagonal_frac=0.0))
table = build_projection_table(ts); R = Renderer(ts, table)

op = np.load(OUT / "opacity.npy")
pred = R.render_np(op)
op_dith = np.stack([halftone.error_diffusion(op[i]).astype(np.float32) for i in range(op.shape[0])])
pred_dith = R.render_np(op_dith)                              # what the DITHERED (cuttable) piece casts
print(f"opacity: mean {op.mean():.3f}  dithered coverage {op_dith.mean():.3f}")

fig, ax = plt.subplots(2, 3, figsize=(13, 10))
for ri, w in enumerate(("A", "B")):
    for ci, (title, d) in enumerate([("TARGET", tg[w]), ("OPTIMIZER (ideal)", pred[w]),
                                     ("FABRICABLE (dithered)", pred_dith[w])]):
        ax[ri, ci].imshow(np.clip(d, 0, 1), origin="lower", cmap="gray_r")
        ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
        if ri == 0:
            ax[ri, ci].set_title(title, fontweight="bold")
    ax[ri, 0].set_ylabel(f"Wall {w}" + ("  (Scream)" if w == "A" else "  (Munch)"), fontsize=12)
plt.suptitle(f"The Scream - grayscale optimizer (seed {SEED}); fabricable = dithered dots", y=0.995, fontsize=13)
plt.tight_layout(); plt.savefig(OUT / "walls.png", dpi=110, bbox_inches="tight"); plt.close()


def save(arr, path, scale=3):
    a = np.clip(1.0 - np.flipud(np.clip(arr, 0, 1)), 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8)); im.resize((im.width * scale, im.height * scale), Image.NEAREST).save(path)
for w in ("A", "B"):
    save(pred_dith[w], OUT / f"recon{w}_fabricable.png")

try:
    build_interactive(ts, table, op, pred, str(OUT / "scene.html"), rays=40, auto_open=False)
    print("wrote scene.html")
except Exception as e:
    print(f"scene.html skipped: {e}")

# fabrication from the DITHERED opacity (tone -> dots); raster_to_pieces still drops sub-min-feature dots
ts_dith = ts
pieces, total = raster_to_pieces(ts_dith, op_dith)
print(f"{total} cut pieces across {len(ts.panels)} panels (after min-feature {ts.fab.min_feature*1000:.0f} mm floor)")
drawings = build_panel_drawings(ts_dith, pieces)
placements = nesting.nest(drawings, ts.fab.sheet_size)
oversize = [p.name for p in placements if p.oversize]
if oversize:
    print(f"WARNING oversize panels: {oversize}")
written = []
for ln, dl in layers.group_mono(drawings).items():
    written += export_dxf.export_all_dxf(dl, OUT / "cut" / ln) + export_svg.export_all_svg(dl, OUT / "cut" / ln)
print(f"wrote {len(written)} cut files under {OUT/'cut'}")
