"""
CONTROLLED test: does segmentation actually help, once the comparison is fair?

My earlier "it doesn't help" was confounded three ways: colour_blend differed between arms, the
means were over different seed sets, and -- the big one -- the region path SKIPPED the shard
budget, so region runs carried ~15% more shards, which this codebase measures as *lower*
SSIM/edge-fidelity on its own. That penalised segmentation for a wiring mistake, not for its
quality. `_autotune_regions` now budget-matches the region path.

Arms (identical seeds, identical colour_blend, budget-matched):
  A uniform     no segmentation -- the true baseline
  B hard        region_masks: no shard may cross an object boundary
  C soft        semantic_masks: object boundaries/parts steer shard DENSITY, tiling stays free
                (this path existed all along and was never tested)
  D hard+soft   both

  py out_thickness_test/seg_ablation.py PAIR [n_seeds] [blend]
     PAIR: girl | vase
"""
import sys, os, dataclasses, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose, search
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
import semseg as SS

PAIR = sys.argv[1] if len(sys.argv) > 1 else "girl"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BLEND = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0     # SAME for every arm
MATCH_TOL, SEM_W = 0.30, 0.6
OUT = "out_thickness_test/mona_pairs"

PAIRS = {"girl": ("examples/girl_front_nobg.png", "examples/girl_back_nobg.png", "face", "sam"),
         "vase": ("examples/sunflowers_clean_nobg.png", "examples/vg_p_yellow_nobg.png", "sam", "face")}
A_IMG, B_IMG, SEG_A, SEG_B = PAIRS[PAIR]

scene0 = load_scene("scenes/tabletop60.yaml"); WR = scene0.solve.wall_res
NAMES = ["clear"] + C.CMYK
SP = dataclasses.replace(scene0.solve, diagonal_frac=0.0)
targets = {"A": C.load_color_target(A_IMG, WR, white_thr=scene0.white_threshold),
           "B": C.load_color_target(B_IMG, WR, white_thr=scene0.white_threshold)}

print(f"pair={PAIR}  seeds=1..{N}  colour_blend={BLEND} (same for all arms)  "
      f"budget={scene0.overlap_shard_budget}\n")
labels, infos, semmaps = {}, {}, {}
for w, kind in (("A", SEG_A), ("B", SEG_B)):
    subj = C.subject_mask(targets[w], scene0.white_threshold)
    if kind == "sam":
        import sam_seg
        labels[w] = sam_seg.masks_to_labels(sam_seg.sam_masks(targets[w]), subj)
        infos[w] = {v: "sam" for v in range(1, int(labels[w].max()) + 1)}
    else:
        labels[w], infos[w] = SS.to_regions(targets[w], subj, kind=kind)
    semmaps[w] = SS.importance_map(labels[w], infos[w], mode="both")
    print(f"  Wall {w} [{kind}]: {int(labels[w].max())} regions, "
          f"soft-map covers {100*float((semmaps[w]>0).mean()):.0f}% of canvas")

ARMS = {"A uniform":  dict(region_masks=None, semantic_masks=None, semantic_weight=0.0),
        "B hard":     dict(region_masks=labels, semantic_masks=None, semantic_weight=0.0),
        "C soft":     dict(region_masks=None, semantic_masks=semmaps, semantic_weight=SEM_W),
        "D hard+soft": dict(region_masks=labels, semantic_masks=semmaps, semantic_weight=SEM_W)}


def run(seed, kw):
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
        match_tol=MATCH_TOL, colour_blend=BLEND, **kw)
    pred = renderer.render_color_np(C.stack_transmit_lut(NAMES, sc, si))
    acc = _metrics.evaluate_wall_accuracy(targets, pred)
    prim = {p.name: primary_wall_of(ts, table, p) for p in ts.panels}
    duty, bleed = search.colour_agreeing_duty(renderer, pred and C.stack_transmit_lut(NAMES, sc, si),
                                              ts.panels, targets, ts.white_threshold,
                                              prim=prim, match_tol=MATCH_TOL)
    return dict(pred=pred, clarity=acc["A"]["ssim"] + acc["B"]["ssim"],
                rmse=0.5 * (acc["A"]["rmse"] + acc["B"]["rmse"]),
                shards=bs.get("A", {}).get("achieved", 0) + bs.get("B", {}).get("achieved", 0),
                dutyB=duty["B"], bleedB=bleed["B"])


res = {}
print(f"\n{'arm':12s} {'clarity':>16} {'shards':>8} {'duty B':>8} {'bleed B':>8}")
print("-" * 58)
for name, kw in ARMS.items():
    runs = [run(s, kw) for s in range(1, N + 1)]
    cl = np.array([r["clarity"] for r in runs]); sh = np.array([r["shards"] for r in runs])
    best = min(runs, key=lambda r: r["rmse"])
    res[name] = dict(mean=float(cl.mean()), std=float(cl.std()), shards=float(sh.mean()),
                     best=float(best["clarity"]), pred=best["pred"],
                     dutyB=float(np.mean([r["dutyB"] for r in runs])),
                     bleedB=float(np.mean([r["bleedB"] for r in runs])))
    r = res[name]
    print(f"{name:12s} {r['mean']:8.3f}+-{r['std']:.3f} {r['shards']:8.0f} "
          f"{r['dutyB']:7.1f}% {r['bleedB']:7.1f}%")

base = res["A uniform"]["mean"]
print(f"\nvs uniform baseline ({base:.3f}):")
for name, r in res.items():
    if name == "A uniform":
        continue
    d = r["mean"] - base
    sig = "SIGNIFICANT" if abs(d) > 2 * max(r["std"], res["A uniform"]["std"]) else "within noise"
    print(f"  {name:12s} {d:+.4f}   ({sig})")

fig, ax = plt.subplots(len(res), 2, figsize=(8.5, 4.1 * len(res)))
for i, (name, r) in enumerate(res.items()):
    for j, w in enumerate(("A", "B")):
        ax[i, j].imshow(np.clip(r["pred"][w], 0, 1), origin="lower", aspect="auto")
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0:
            ax[i, j].set_title(f"Wall {w}", fontsize=12, fontweight="bold")
    ax[i, 0].set_ylabel(f"{name}\n{r['mean']:.3f}  {r['shards']:.0f} shards", fontsize=10)
plt.suptitle(f"Segmentation ablation ({PAIR}) — same seeds, same blend, budget-matched", fontsize=13)
plt.tight_layout(); plt.savefig(f"{OUT}/seg_ablation_{PAIR}.png", dpi=100, bbox_inches="tight")
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in res.items()},
          open(f"{OUT}/seg_ablation_{PAIR}.json", "w"), indent=2)
print(f"\nsaved {OUT}/seg_ablation_{PAIR}.png / .json")
