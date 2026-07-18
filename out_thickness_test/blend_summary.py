"""
ONE image: what the joint (depth+colour) shard selection changed.

Per pair, the SAME seed is solved twice -- colour_blend OFF (today) and ON (0.6) -- so the only
difference is the feature. Shows both walls for each arm plus the measured deltas.

Run:  py out_thickness_test/blend_summary.py
"""
import sys, os, dataclasses, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose, search
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics

BLEND_ON = 0.6
MATCH_TOL = 0.30
OUT = "out_thickness_test/mona_pairs"
PAIRS = [
    ("cut sunflowers x oranges", "examples/sf_surface_nobg.png", "examples/oranges_nobg.png", 1),
    ("vase x oranges",           "examples/sunflowers_clean_nobg.png", "examples/oranges_nobg.png", 4),
    ("vase x cut sunflowers",    "examples/sunflowers_clean_nobg.png", "examples/sf_surface_nobg.png", 1),
]
scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res
NAMES = ["clear"] + C.CMYK
SP = dataclasses.replace(scene0.solve, diagonal_frac=0.0)
AGG = json.load(open(f"{OUT}/blend_compare.json"))     # 6-seed means, the trustworthy stats


def run(targets, seed, blend):
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
        match_tol=MATCH_TOL, colour_blend=blend)
    pT = C.stack_transmit_lut(NAMES, sc, si)
    pred = renderer.render_color_np(pT)
    acc = _metrics.evaluate_wall_accuracy(targets, pred)
    prim = {p.name: primary_wall_of(ts, table, p) for p in ts.panels}
    duty, bleed = search.colour_agreeing_duty(renderer, pT, ts.panels, targets,
                                              ts.white_threshold, prim=prim, match_tol=MATCH_TOL)
    return pred, acc, duty, bleed


rows = []
for label, pa, pb, seed in PAIRS:
    targets = {"A": C.load_color_target(pa, WR, white_thr=scene0.white_threshold),
               "B": C.load_color_target(pb, WR, white_thr=scene0.white_threshold)}
    print(f"solving {label} (seed {seed}) OFF/ON ...")
    off = run(targets, seed, 0.0)
    on = run(targets, seed, BLEND_ON)
    rows.append((label, targets, off, on, AGG[label]))

fig, ax = plt.subplots(len(rows), 5, figsize=(19, 4.0 * len(rows)),
                       gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.95]})
if len(rows) == 1:
    ax = ax[None, :]
for i, (label, targets, off, on, agg) in enumerate(rows):
    panels_ = [(off[0], "A", "OFF (today)", "#b03030"), (off[0], "B", "OFF (today)", "#b03030"),
               (on[0], "A", f"ON blend {BLEND_ON:g}", "#207040"),
               (on[0], "B", f"ON blend {BLEND_ON:g}", "#207040")]
    for ci, (pred, wall, tag, colr) in enumerate(panels_):
        ax[i, ci].imshow(np.clip(pred[wall], 0, 1), origin="lower", aspect="auto")
        ax[i, ci].set_xticks([]); ax[i, ci].set_yticks([])
        if i == 0:
            ax[i, ci].set_title(f"{tag} — Wall {wall}", fontsize=11, fontweight="bold", color=colr)
    a0, a1 = agg["0.0"], agg[str(BLEND_ON)]
    dB = a1["dutyB"] - a0["dutyB"]; bB = a1["bleedB"] - a0["bleedB"]
    dcl = a1["clarity_mean"] - a0["clarity_mean"]
    ax[i, 4].axis("off")
    txt = (f"6-seed means\n\n"
           f"clarity   {a0['clarity_mean']:.3f} -> {a1['clarity_mean']:.3f}\n"
           f"          ({dcl:+.3f})  unchanged\n\n"
           f"duty  B   {a0['dutyB']:.1f}% -> {a1['dutyB']:.1f}%\n"
           f"          ({dB:+.1f}) more shards\n"
           f"          serving BOTH walls\n\n"
           f"bleed B   {a0['bleedB']:.1f}% -> {a1['bleedB']:.1f}%\n"
           f"          ({bB:+.1f}) less wrong-\n"
           f"          colour contamination")
    ax[i, 4].text(0.0, 0.5, txt, fontsize=10, va="center", family="monospace")
    ax[i, 0].set_ylabel(label, fontsize=10)

plt.suptitle("Joint (depth + colour) shard selection — same seed, OFF vs ON.  "
             "Picture unchanged; more genuine double duty, less wrong-colour bleed.",
             fontsize=13, y=0.999)
plt.tight_layout()
plt.savefig(f"{OUT}/blend_summary.png", dpi=105, bbox_inches="tight"); plt.close()
print(f"\nsaved {OUT}/blend_summary.png")
