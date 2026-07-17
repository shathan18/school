"""
FULL-QUALITY DELIVERABLE on the best-compatibility pair (Monet day/dusk, edited).
Current FIXED pipeline: signed-damage dw0.5/cw0.5, colour-agreement credit (match_tol=0.30).

Multi-seed (mandatory) -> pick best by mean SSIM -> full render + interactive HTML + all
metrics including the CORRECTED colour-agreeing B_good (naive "lands on subject" retired).
"""
import sys, dataclasses, os, base64, json
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
from shadowart.preview.interactive3d import build_interactive

A_IMG = "examples/monet_day_edited.jpeg"
B_IMG = "examples/monet_dusk_edited.jpeg"
MATCH_TOL = 0.30
OUT = "out_thickness_test/monet_final"; os.makedirs(OUT, exist_ok=True)

scene = load_scene("scenes/example.yaml"); wr = scene.solve.wall_res
names = ["clear"] + C.CMYK
targets = {"A": C.load_color_target(A_IMG, wr, white_thr=scene.white_threshold),
           "B": C.load_color_target(B_IMG, wr, white_thr=scene.white_threshold)}
subject = {w: C.subject_mask(targets[w], scene.white_threshold) for w in ("A", "B")}
SP = scene.solve   # pipeline default density (1x); the 0.5x used earlier was a cross-talk-sweep artifact


def colour_agree_bgood(pred, renderer, pT, panels, prim):
    """Honest colour-agreeing B_good: % of secondary subject whose stray coverage renders a
    colour matching the target there (||pred-tgt||<match_tol). The ONLY B_good reported."""
    def render_keep(keep):
        q = pT.copy()
        for gi, p in enumerate(panels):
            if p.name not in keep:
                q[gi] = 1.0
        return renderer.render_color_np(q)
    out = {}
    for w in ("A", "B"):
        subj = subject[w]; denom = max(subj.sum(), 1)
        nonprim = {p.name for p in panels if prim[p.name] != w}
        xr = render_keep(nonprim)[w]
        onsub = ((1.0 - xr.mean(-1)) > 0.05) & subj
        if onsub.any():
            d = np.sqrt(((xr[onsub] - targets[w][onsub]) ** 2).sum(1))
            out[w] = 100.0 * (d < MATCH_TOL).sum() / denom
        else:
            out[w] = 0.0
    return out


def run(seed):
    panels, _ = build_panels_greedy(scene, count=14, mode="deliberate", K=16,
                                    targets=targets, seed=seed, angle_deg_range=(5, 85))
    ts = dataclasses.replace(scene, panels=panels, solve=SP)
    table = build_projection_table(ts); renderer = Renderer(ts, table)
    sc, op, fr, res, sd, bs, si = decompose.fragment_shards_overlap(
        ts, table, targets, names=names, white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack, seed=seed, damage_weight=0.5, credit_weight=0.5,
        match_tol=MATCH_TOL)
    pT = C.stack_transmit_lut(names, sc, si)
    pred = renderer.render_color_np(pT)
    acc = _metrics.evaluate_wall_accuracy(targets, pred)
    prim = {p.name: primary_wall_of(ts, table, p) for p in panels}
    bg = colour_agree_bgood(pred, renderer, pT, panels, prim)
    shards = bs.get("A", {}).get("achieved", 0) + bs.get("B", {}).get("achieved", 0)
    return dict(seed=seed, pred=pred, acc=acc, bg=bg, shards=shards, used=len({f["panel"] for f in fr}),
                ts=ts, table=table, op=op, sc=sc, si=si, panels=panels)


SEEDS = [1, 2, 3, 4, 5]
print(f"pair: Monet day/dusk (edited)   |   pipeline: signed dw0.5 cw0.5, colour-agree credit (match_tol {MATCH_TOL})\n")
print(f"{'seed':>4} {'shards':>7} {'used':>5} | {'A rmse/ssim/edge':>22} | {'B rmse/ssim/edge':>22} | {'B_good A/B (honest)':>20}")
print("-" * 96)
runs = []
for s in SEEDS:
    r = run(s); runs.append(r)
    a, b = r["acc"]["A"], r["acc"]["B"]
    print(f"{s:>4} {r['shards']:>7} {r['used']:>4}/14 | "
          f"{a['rmse']:.3f}/{a['ssim']:.3f}/{a['edge_fidelity']:.3f}      | "
          f"{b['rmse']:.3f}/{b['ssim']:.3f}/{b['edge_fidelity']:.3f}      | "
          f"{r['bg']['A']:5.1f}% /{r['bg']['B']:5.1f}%")

meanrmse = lambda r: 0.5 * (r["acc"]["A"]["rmse"] + r["acc"]["B"]["rmse"])  # lower = better overall fidelity
best = min(runs, key=meanrmse)
ssA = [r["acc"]["A"]["ssim"] for r in runs]; ssB = [r["acc"]["B"]["ssim"] for r in runs]
bgA = [r["bg"]["A"] for r in runs]; bgB = [r["bg"]["B"] for r in runs]
print(f"\nacross {len(SEEDS)} seeds:  SSIM_A {np.mean(ssA):.3f}±{np.std(ssA):.3f}  SSIM_B {np.mean(ssB):.3f}±{np.std(ssB):.3f}"
      f"  |  B_good_A {np.mean(bgA):.1f}±{np.std(bgA):.1f}%  B_good_B {np.mean(bgB):.1f}±{np.std(bgB):.1f}%")
print(f"best seed by mean RMSE = {best['seed']}")

# ---- full-quality rendered walls (target vs reconstruction) ----
r = best; pred = r["pred"]
fig, ax = plt.subplots(2, 2, figsize=(11, 11))
for ri, w in enumerate(("A", "B")):
    ax[ri, 0].imshow(np.clip(targets[w], 0, 1), origin="lower", aspect="auto")
    ax[ri, 1].imshow(np.clip(pred[w], 0, 1), origin="lower", aspect="auto")
    for ci in (0, 1):
        ax[ri, ci].set_xticks([]); ax[ri, ci].set_yticks([])
    ax[ri, 0].set_ylabel(["Wall A  (Monet, day)", "Wall B  (Monet, dusk)"][ri], fontsize=13)
ax[0, 0].set_title("SOURCE", fontsize=14, fontweight="bold")
ax[0, 1].set_title("RECONSTRUCTED SHADOW", fontsize=14, fontweight="bold")
plt.suptitle(f"Monet day/dusk — full pipeline, seed {r['seed']} ({r['shards']} shards, {r['used']}/14 panels)",
             fontsize=15, y=0.995)
plt.tight_layout()
plt.savefig(f"{OUT}/walls.png", dpi=110, bbox_inches="tight"); plt.close()
print(f"\nsaved {OUT}/walls.png")

# ---- interactive 3D scene (local, self-contained plotly) ----
ts, table = r["ts"], r["table"]
sp_pieces = decompose.panel_stack_pieces(ts, r["sc"], names)
pc = {id(p): ch for items in sp_pieces.values() for p, ch, _s in items}
flat = {n: [p for p, _c, _s in items] for n, items in sp_pieces.items()}
col = lambda panel, poly: tuple(C.display_rgb(pc.get(id(poly), "clear")))
html, _ = build_interactive(ts, table, r["op"], None, f"{OUT}/scene.html",
                            rays=40, auto_open=False, wall_rgb=pred,
                            pieces=flat, color_of=col, embed_plotly=True)
print(f"wrote {html}")

# ---- dump metrics json for the artifact page ----
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump({"best_seed": r["seed"], "shards": r["shards"], "panels_used": r["used"],
               "per_seed": [{"seed": x["seed"], "A": x["acc"]["A"], "B": x["acc"]["B"], "bg": x["bg"]} for x in runs],
               "mean": {"ssimA": float(np.mean(ssA)), "ssimB": float(np.mean(ssB)),
                        "bgA": float(np.mean(bgA)), "bgB": float(np.mean(bgB)),
                        "bgA_sd": float(np.std(bgA)), "bgB_sd": float(np.std(bgB))}}, f, indent=2)
print(f"wrote {OUT}/metrics.json")
