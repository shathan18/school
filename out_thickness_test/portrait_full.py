"""
Both walls, all five options — the whole piece as it would appear, for a human judgement call.

Row = one candidate pairing (sunflower vase on Wall A, a Van Gogh portrait on Wall B).
Columns: vase source | vase shadow | portrait source | portrait shadow.

Judge two things at once: does the portrait read as Van Gogh, and does the vase stay clean?

Run:  py out_thickness_test/portrait_full.py
"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

RUNS = [("blue (with palette)", "out_thickness_test/sam_vase_blue"),
        ("gold (1887)", "out_thickness_test/sem_vase_gold"),
        ("generated", "out_thickness_test/sam_vase_gemini"),
        ("yellow", "out_thickness_test/sam_vase_yellow"),
        ("dark (pointillist)", "out_thickness_test/sam_vase_dark")]
OUT = "out_thickness_test/mona_pairs"
COLS = [("srcA.png", "vase — source"), ("reconA.png", "vase — SHADOW"),
        ("srcB.png", "portrait — source"), ("reconB.png", "portrait — SHADOW")]

rows = []
for label, d in RUNS:
    p = os.path.join(d, "metrics.json")
    if os.path.exists(p):
        m = json.load(open(p))
        rows.append((label, d, m.get("clarity_mean", 0.0), m.get("shards", 0)))
rows.sort(key=lambda r: -r[2])

fig, ax = plt.subplots(len(rows), 4, figsize=(16, 4.1 * len(rows)))
for i, (label, d, clarity, shards) in enumerate(rows):
    for j, (fn, title) in enumerate(COLS):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            ax[i, j].imshow(np.asarray(Image.open(p)))
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0:
            ax[i, j].set_title(title, fontsize=12, fontweight="bold")
    ax[i, 0].set_ylabel(f"{label}\nclarity {clarity:.3f}\n{shards} shards", fontsize=10)
plt.suptitle("All five candidates, both walls — your call", fontsize=15, y=0.998)
plt.tight_layout()
plt.savefig(f"{OUT}/portrait_full.png", dpi=100, bbox_inches="tight")
print(f"saved {OUT}/portrait_full.png")
for label, d, c, sh in rows:
    print(f"  {label:22s} clarity {c:.3f}  {sh:>3} shards   {d}/scene.html")
