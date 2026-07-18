"""
ALL 12 object x portrait results in one image.

vg_matrix.py only saved visuals for the top 4, so this re-solves each pair at the best seed it
recorded in vg_matrix.json (12 solves, not 72) and lays every reconstruction out as a grid:
rows = object, columns = the four Van Gogh self-portraits.

Each cell shows the pair's two reconstructed walls side by side (object shadow | portrait shadow)
with clarity and double duty underneath.

Run:  py out_thickness_test/vg_matrix_all.py
"""
import sys, os, dataclasses, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics

BLEND, MATCH_TOL = 0.6, 0.30
OUT = "out_thickness_test/mona_pairs"
OBJECTS = [("cut sunflowers", "examples/sf_surface_nobg.png"),
           ("oranges",        "examples/oranges_nobg.png"),
           ("sunflower vase", "examples/sunflowers_clean_nobg.png")]
PORTRAITS = [("gold",   "examples/vg_p_gold_nobg.png"),
             ("yellow", "examples/vg_p_yellow_nobg.png"),
             ("dark",   "examples/vg_p_dark_nobg.png"),
             ("blue",   "examples/vg_p_blue_nobg.png")]
scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res
NAMES = ["clear"] + C.CMYK
SP = dataclasses.replace(scene0.solve, diagonal_frac=0.0)
REC = {(r["obj"], r["por"]): r for r in json.load(open(f"{OUT}/vg_matrix.json"))}


def solve(targets, seed):
    panels, _ = build_panels_greedy(scene0, count=14, mode="deliberate", K=16, targets=targets,
                                    seed=seed, angle_deg_range=(5, 85),
                                    anchor_range=SP.search_anchor_range, standoff=SP.search_standoff,
                                    mag_cap=SP.search_mag_cap, u_size_range=SP.search_u_size_range,
                                    v_range=SP.search_v_range)
    ts = dataclasses.replace(scene0, panels=panels, solve=SP)
    table = build_projection_table(ts); renderer = Renderer(ts, table)
    sc, op, fr, rs, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, targets, names=NAMES, white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack, seed=seed, damage_weight=0.5, credit_weight=0.5,
        match_tol=MATCH_TOL, colour_blend=BLEND)
    return renderer.render_color_np(C.stack_transmit_lut(NAMES, sc, si))


def side_by_side(a, b, gap=8):
    """Two wall images into one canvas: object shadow | portrait shadow."""
    a = np.clip(a, 0, 1); b = np.clip(b, 0, 1)
    h = max(a.shape[0], b.shape[0])
    def pad(x):
        out = np.ones((h, x.shape[1], 3))
        out[:x.shape[0]] = x
        return out
    return np.concatenate([pad(a), np.ones((h, gap, 3)), pad(b)], axis=1)


fig, ax = plt.subplots(len(OBJECTS), len(PORTRAITS) + 1,
                       figsize=(4.4 * (len(PORTRAITS) + 1), 4.0 * len(OBJECTS)),
                       gridspec_kw={"width_ratios": [0.75] + [1] * len(PORTRAITS)})
for i, (oname, opath) in enumerate(OBJECTS):
    tA = C.load_color_target(opath, WR, white_thr=scene0.white_threshold)
    ax[i, 0].imshow(np.clip(tA, 0, 1), origin="lower", aspect="auto")
    ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
    ax[i, 0].set_ylabel(oname, fontsize=12, fontweight="bold")
    if i == 0:
        ax[i, 0].set_title("object (source)", fontsize=11)
    for j, (pname, ppath) in enumerate(PORTRAITS):
        r = REC[(oname, pname)]
        targets = {"A": tA, "B": C.load_color_target(ppath, WR, white_thr=scene0.white_threshold)}
        print(f"rendering {oname} x {pname} (seed {r['best_seed']}) ...")
        pred = solve(targets, r["best_seed"])
        ax[i, j + 1].imshow(side_by_side(pred["A"], pred["B"]), origin="lower", aspect="auto")
        ax[i, j + 1].set_xticks([]); ax[i, j + 1].set_yticks([])
        if i == 0:
            ax[i, j + 1].set_title(f"x  {pname} portrait", fontsize=12, fontweight="bold")
        ax[i, j + 1].set_xlabel(f"clarity {r['clarity_mean']:.3f}   "
                                f"duty {r['dutyA']:.0f}/{r['dutyB']:.0f}%", fontsize=10)
plt.suptitle("All 12 pairings — Van Gogh object (Wall A) x self-portrait (Wall B).  "
             "Each cell: object shadow | portrait shadow.", fontsize=14, y=0.999)
plt.tight_layout()
plt.savefig(f"{OUT}/vg_matrix_all.png", dpi=100, bbox_inches="tight"); plt.close()
print(f"\nsaved {OUT}/vg_matrix_all.png")
