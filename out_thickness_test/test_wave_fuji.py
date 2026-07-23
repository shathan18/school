"""Test the Hokusai pair (Great Wave x Red Fuji) through the colour-overlap pipeline: report the
honest colour-agreeing double-duty (B_good, same metric ceiling_straddle.py reports), against an
ARBITRARY pair as the floor, and render the reconstruction."""
import sys, os, dataclasses, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose, search
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics

scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res
NAMES = ["clear"] + C.CMYK
MATCH_TOL = 0.30
sp = scene0.solve


def test(pa, pb, seed=2):
    t = {"A": C.load_color_target(pa, WR, white_thr=scene0.white_threshold),
         "B": C.load_color_target(pb, WR, white_thr=scene0.white_threshold)}
    panels, _ = build_panels_greedy(scene0, count=14, mode="deliberate", K=16, targets=t, seed=seed,
                                    angle_deg_range=(5, 85), anchor_range=sp.search_anchor_range,
                                    standoff=sp.search_standoff, mag_cap=sp.search_mag_cap,
                                    u_size_range=sp.search_u_size_range, v_range=sp.search_v_range)
    ts = dataclasses.replace(scene0, panels=panels, solve=dataclasses.replace(sp, diagonal_frac=0.0))
    table = build_projection_table(ts); R = Renderer(ts, table)
    sc, op, fr, rs, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, t, names=NAMES, white_thr=ts.white_threshold, max_stack=ts.color_max_stack,
        seed=seed, damage_weight=0.5, credit_weight=0.5, match_tol=MATCH_TOL)
    pT = C.stack_transmit_lut(NAMES, sc, si); pred = R.render_color_np(pT)
    prim = {p.name: primary_wall_of(ts, table, p) for p in ts.panels}
    good, bad = search.colour_agreeing_duty(R, pT, ts.panels, t, ts.white_threshold, prim=prim, match_tol=MATCH_TOL)
    acc = _metrics.evaluate_wall_accuracy(t, pred)
    n = int(bs.get("A", {}).get("achieved", 0)) + int(bs.get("B", {}).get("achieved", 0))
    return t, pred, good, bad, acc, n


import sys as _sys
PAIR_B = _sys.argv[1] if len(_sys.argv) > 1 else "examples/red_fuji.jpg"
OUT_DIR = _sys.argv[2] if len(_sys.argv) > 2 else "out_wavefuji"

print(f"running Hokusai pair (Great Wave x {PAIR_B}) ...")
tW, predW, goodW, badW, accW, nW = test("examples/wave_src.jpg", PAIR_B)
print("running ARBITRARY floor (apples x breakfast) ...")
tC, predC, goodC, badC, accC, nC = test("examples/apples.jpg", "examples/breakfast.jpg")


def line(name, good, bad, acc, n):
    mg = 0.5 * (good["A"] + good["B"]); mb = 0.5 * (bad["A"] + bad["B"])
    print(f"\n{name}  ({n} shards)")
    print(f"  B_good (colour-agreeing double duty): A {good['A']:.1f}%  B {good['B']:.1f}%  | mean {mg:.1f}%")
    print(f"  bleed (wrong-colour cross-talk):      A {bad['A']:.1f}%  B {bad['B']:.1f}%  | mean {mb:.1f}%")
    print(f"  wall SSIM: A {acc['A']['ssim']:.3f}  B {acc['B']['ssim']:.3f}")
    return mg


mgW = line("HOKUSAI  Great Wave x Red Fuji  (compatible?)", goodW, badW, accW, nW)
mgC = line("ARBITRARY  apples x breakfast", goodC, badC, accC, nC)
print(f"\n==> Hokusai mean B_good {mgW:.1f}%  vs  arbitrary floor {mgC:.1f}%   "
      f"(target compatibility band ~15-25%)")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for ri, w in enumerate(("A", "B")):
    ax[ri, 0].imshow(np.clip(tW[w], 0, 1), origin="lower", aspect="auto")
    ax[ri, 1].imshow(np.clip(predW[w], 0, 1), origin="lower", aspect="auto")
    for ci in (0, 1): ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
    ax[ri, 0].set_ylabel(f"Wall {w}", fontsize=12)
ax[0, 0].set_title("SOURCE", fontweight="bold"); ax[0, 1].set_title("RECONSTRUCTION", fontweight="bold")
plt.suptitle(f"Hokusai: Great Wave x {os.path.basename(PAIR_B)}  -  B_good {mgW:.1f}% "
             f"(arbitrary floor {mgC:.1f}%), {nW} shards", fontsize=13, y=0.99)
plt.tight_layout(); os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(f"{OUT_DIR}/walls.png", dpi=110, bbox_inches="tight")
print(f"saved {OUT_DIR}/walls.png")
