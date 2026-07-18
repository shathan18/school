"""
Was the scene model good enough? Bake-off between the tiny SegFormer-B0 I used and stronger
scene parsers, on the actual paintings.

B0 is the SMALLEST SegFormer (~3.7M params). Its output on these images was suspect: it called
the vase a "lamp", invented a "wall" on a plain white background, and labelled all of Starry
Night "painting". This compares it against B5 and Mask2Former-Swin-L (SOTA-class on ADE20K) to
separate "the model is weak" from "photo-trained models can't read Van Gogh".

Run:  py out_thickness_test/model_bakeoff.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from PIL import Image
from collections import Counter
from shadowart.config.io import load_scene
from shadowart.targets import color as C
import objectseg as OS

MODELS = [("B0 (used)", "nvidia/segformer-b0-finetuned-ade-512-512", "segformer"),
          ("B5", "nvidia/segformer-b5-finetuned-ade-640-640", "segformer"),
          ("Mask2Former-L", "facebook/mask2former-swin-large-ade-semantic", "mask2former")]
IMAGES = [("vase", "examples/sunflowers_clean_nobg.png"),
          ("starry night", "examples/starry_night.png"),
          ("oranges", "examples/oranges_nobg.png")]

sc = load_scene("scenes/tabletop60.yaml"); WR = sc.solve.wall_res


def predict(rgb, name, kind):
    from transformers import AutoImageProcessor, SegformerImageProcessor
    H, W = rgb.shape[:2]
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    try:
        proc = AutoImageProcessor.from_pretrained(name)
    except Exception:
        proc = SegformerImageProcessor()
    if kind == "mask2former":
        from transformers import Mask2FormerForUniversalSegmentation as M
        model = M.from_pretrained(name).eval()
        with torch.no_grad():
            out = model(**proc(images=img, return_tensors="pt"))
        seg = proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0].cpu().numpy()
        id2l = model.config.id2label
    else:
        from transformers import AutoModelForSemanticSegmentation as M
        model = M.from_pretrained(name).eval()
        with torch.no_grad():
            logits = model(**proc(images=img, return_tensors="pt")).logits
        up = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear",
                                             align_corners=False)
        seg = up.argmax(1)[0].cpu().numpy()
        id2l = model.config.id2label
    return seg.astype(int), {int(k): v for k, v in id2l.items()}


fig, ax = plt.subplots(len(IMAGES), len(MODELS) + 1, figsize=(4.3 * (len(MODELS) + 1), 4.0 * len(IMAGES)))
for i, (iname, path) in enumerate(IMAGES):
    t = C.load_color_target(path, WR, white_thr=sc.white_threshold)
    m = C.subject_mask(t, sc.white_threshold)
    ax[i, 0].imshow(np.clip(t, 0, 1), origin="lower"); ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
    ax[i, 0].set_ylabel(iname, fontsize=12, fontweight="bold")
    if i == 0:
        ax[i, 0].set_title("source", fontsize=11)
    for j, (label, name, kind) in enumerate(MODELS):
        try:
            seg, id2l = predict(t, name, kind)
        except Exception as e:
            print(f"{iname:14s} {label:14s} FAILED {type(e).__name__}: {e}")
            ax[i, j + 1].axis("off"); continue
        inside = seg[m]
        cnt = Counter(id2l.get(int(c), str(c)) for c in inside)
        top = cnt.most_common(6)
        frac = {k: round(100 * v / max(len(inside), 1)) for k, v in top}
        print(f"{iname:14s} {label:14s} {len(set(inside.tolist())):2d} classes | {frac}")
        lab = np.zeros_like(seg)
        for n, c in enumerate(sorted(set(inside.tolist())), start=1):
            lab[(seg == c) & m] = n
        ax[i, j + 1].imshow(OS.overlay(t, lab), origin="lower")
        ax[i, j + 1].set_xticks([]); ax[i, j + 1].set_yticks([])
        ax[i, j + 1].set_xlabel(", ".join(f"{k}" for k, _ in top[:3]), fontsize=8)
        if i == 0:
            ax[i, j + 1].set_title(label, fontsize=12, fontweight="bold")
plt.suptitle("Scene-parser bake-off on the paintings — is B0 good enough?", fontsize=14, y=0.999)
plt.tight_layout()
plt.savefig("out_thickness_test/mona_pairs/model_bakeoff.png", dpi=100, bbox_inches="tight")
print("\nsaved out_thickness_test/mona_pairs/model_bakeoff.png")
