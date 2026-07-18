"""
Which Van Gogh reads as Van Gogh?

All five portraits through the identical pipeline (SAM instances on the vase wall, face-parsing
+ finer shards on skin/eyes for the portrait wall), paired with the sunflower vase.

The user's criterion is NOT the metric -- it is whether the shadow is unmistakably a reference
to Van Gogh. The clarity spread across these is ~0.05, far too small to decide that, so this
puts the portrait walls side by side at full size for a human call. Source above, shadow below.

Run:  py out_thickness_test/portrait_compare.py
"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

RUNS = [("blue (with palette)", "out_thickness_test/sam_vase_blue", "examples/vg_p_blue_nobg.png"),
        ("gold (1887)", "out_thickness_test/sem_vase_gold", "examples/vg_p_gold_nobg.png"),
        ("generated", "out_thickness_test/sam_vase_gemini", "examples/vg_p_gemini_nobg.png"),
        ("yellow", "out_thickness_test/sam_vase_yellow", "examples/vg_p_yellow_nobg.png"),
        ("dark (pointillist)", "out_thickness_test/sam_vase_dark", "examples/vg_p_dark_nobg.png")]
OUT = "out_thickness_test/mona_pairs"

rows = []
for label, d, src in RUNS:
    p = os.path.join(d, "metrics.json")
    if not os.path.exists(p):
        print(f"  MISSING {d}"); continue
    m = json.load(open(p))
    rows.append((label, d, src, m.get("clarity_mean", 0.0), m.get("shards", 0)))
rows.sort(key=lambda r: -r[3])

fig, ax = plt.subplots(2, len(rows), figsize=(3.5 * len(rows), 8.4))
for i, (label, d, src, clarity, shards) in enumerate(rows):
    ax[0, i].imshow(np.asarray(Image.open(src)))
    ax[0, i].set_title(f"{label}", fontsize=12, fontweight="bold")
    rec = os.path.join(d, "reconB.png")
    if os.path.exists(rec):
        ax[1, i].imshow(np.asarray(Image.open(rec)))
    ax[1, i].set_xlabel(f"clarity {clarity:.3f}   {shards} shards", fontsize=10)
    for r in (0, 1):
        ax[r, i].set_xticks([]); ax[r, i].set_yticks([])
ax[0, 0].set_ylabel("SOURCE", fontsize=12, fontweight="bold")
ax[1, 0].set_ylabel("CAST SHADOW", fontsize=12, fontweight="bold")
plt.suptitle("Which one still reads as Van Gogh on the wall?  (all paired with the sunflower vase)",
             fontsize=14, y=0.99)
plt.tight_layout()
plt.savefig(f"{OUT}/portrait_compare.png", dpi=105, bbox_inches="tight")
print(f"saved {OUT}/portrait_compare.png\n")
for label, _d, _s, c, sh in rows:
    print(f"  {label:22s} clarity {c:.3f}   {sh} shards")
