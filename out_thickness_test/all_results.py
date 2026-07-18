"""
Everything in ONE image: the three Van Gogh pairings, both walls, source beside reconstruction,
with metrics. Row per pairing, columns = [Wall A src | Wall A recon | Wall B src | Wall B recon].

Run:  py out_thickness_test/all_results.py
"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

RUNS = [("cut sunflowers  x  oranges", "out_thickness_test/mix_surface_oranges"),
        ("sunflower vase  x  oranges", "out_thickness_test/mix_vase_oranges"),
        ("sunflower vase  x  cut sunflowers", "out_thickness_test/mix_vase_surface")]
OUT = "out_thickness_test/mona_pairs"; os.makedirs(OUT, exist_ok=True)
DST = f"{OUT}/all_results.png"

rows = []
for label, d in RUNS:
    p = os.path.join(d, "metrics.json")
    if not os.path.exists(p):
        print(f"  MISSING {d} -- skipped"); continue
    with open(p) as f:
        m = json.load(f)
    mean = m["mean"]
    rows.append(dict(label=label, dir=d, ssimA=mean["ssimA"], ssimB=mean["ssimB"],
                     clarity=mean["ssimA"] + mean["ssimB"],
                     bgA=mean["bgA"], bgB=mean["bgB"],
                     seed=m["best_seed"], shards=m["shards"]))
rows.sort(key=lambda r: r["clarity"], reverse=True)

COLS = [("srcA.png", "Wall A — source"), ("reconA.png", "Wall A — shadow"),
        ("srcB.png", "Wall B — source"), ("reconB.png", "Wall B — shadow")]
n = len(rows)
fig, ax = plt.subplots(n, 5, figsize=(17, 3.9 * n),
                       gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.62]})
if n == 1:
    ax = ax[None, :]
for i, r in enumerate(rows):
    for ci, (fn, title) in enumerate(COLS):
        p = os.path.join(r["dir"], fn)
        if os.path.exists(p):
            ax[i, ci].imshow(np.asarray(Image.open(p)))
        ax[i, ci].set_xticks([]); ax[i, ci].set_yticks([])
        if i == 0:
            ax[i, ci].set_title(title, fontsize=11, fontweight="bold")
    ax[i, 0].set_ylabel(f"#{i+1}\n{r['label']}", fontsize=10)
    ax[i, 4].axis("off")
    txt = (f"clarity {r['clarity']:.3f}\n\nSSIM A {r['ssimA']:.3f}\n     B {r['ssimB']:.3f}\n\n"
           f"double duty\n  A {r['bgA']:.1f}%\n  B {r['bgB']:.1f}%\n\n"
           f"seed {r['seed']}/100\n{r['shards']} shards")
    ax[i, 4].text(0.0, 0.5, txt, fontsize=10.5, va="center", family="monospace")

plt.suptitle("ShadowArt — Van Gogh still-life pairings on the 60 cm tabletop body  "
             "(100 greedy restarts each, best by RMSE)", fontsize=14, y=0.998)
plt.tight_layout()
plt.savefig(DST, dpi=105, bbox_inches="tight"); plt.close()
print(f"saved {DST}")
for i, r in enumerate(rows, 1):
    print(f"  #{i} {r['label']:36s} clarity {r['clarity']:.3f}")
