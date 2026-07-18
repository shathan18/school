"""
SAM (Segment Anything) -- class-agnostic INSTANCE segmentation.

Why this and not the semantic parsers: SegFormer / Mask2Former label CLASSES. They return
"flower" as one region and "food" as one blob -- they cannot give you *each* flower or *each*
orange. The user's ask ("each flower is an object, each star is an object") is an INSTANCE
problem, which is what SAM is built for: it proposes a mask for every distinct thing it sees,
without naming any of them.

Strategy here: SAM supplies the object boundaries (instances), and the semantic models
(semseg.py) can still be layered on top to NAME those instances and set per-part shard density.

`facebook/sam-vit-huge` on CPU is slow (minutes per image) -- that is expected and accepted.

Run:  py out_thickness_test/sam_seg.py [model] [points_per_side]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from shadowart.config.io import load_scene
from shadowart.targets import color as C
import objectseg as OS

# NB: only read argv when run as a script -- this module is imported by semantic_run.py, which
# has its own (incompatible) argv.
MODEL = "facebook/sam-vit-huge"
PPS = 24
if __name__ == "__main__":
    MODEL = sys.argv[1] if len(sys.argv) > 1 else MODEL
    PPS = int(sys.argv[2]) if len(sys.argv) > 2 else PPS
_PIPE = {}


def _pipe():
    if "p" not in _PIPE:
        from transformers import pipeline
        _PIPE["p"] = pipeline("mask-generation", model=MODEL, device=-1)
    return _PIPE["p"]


def sam_masks(rgb, points_per_side=None, batch=16):
    """List of boolean masks, largest first."""
    pps = points_per_side or PPS
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    out = _pipe()(img, points_per_batch=batch, points_per_crop=pps)
    masks = [np.asarray(m, bool) for m in out["masks"]]
    masks.sort(key=lambda m: -m.sum())
    return masks


def masks_to_labels(masks, subject, min_frac=0.003, max_frac=0.85):
    """Flatten overlapping SAM masks into one label map: paint largest first so smaller,
    nested objects (a single flower inside the bouquet) overwrite the big parent and survive
    as their own region. Then grow the labels over anything left uncovered."""
    H, W = subject.shape
    out = np.zeros((H, W), int)
    total = max(int(subject.sum()), 1)
    nxt = 0
    for m in masks:
        m = m & subject
        f = m.sum() / total
        if f < min_frac or f > max_frac:
            continue
        nxt += 1
        out[m] = nxt
    if nxt == 0:
        return out
    holes = subject & (out == 0)
    if holes.any():
        inds = ndimage.distance_transform_edt(out == 0, return_indices=True, return_distances=False)
        out = np.where(holes, out[tuple(inds)], out)
    out[~subject] = 0
    # renumber contiguously (some ids get fully overwritten)
    uniq = [v for v in np.unique(out) if v != 0]
    remap = {v: i + 1 for i, v in enumerate(uniq)}
    return np.vectorize(lambda v: remap.get(v, 0))(out).astype(int)


if __name__ == "__main__":
    sc = load_scene("scenes/tabletop60.yaml"); WR = sc.solve.wall_res
    IMAGES = [("vase", "examples/sunflowers_clean_nobg.png"),
              ("yellow portrait", "examples/vg_p_yellow_nobg.png"),
              ("oranges", "examples/oranges_nobg.png"),
              ("starry night", "examples/starry_night.png")]
    print(f"model={MODEL}  points_per_side={PPS}  (CPU -- expect minutes per image)\n")
    fig, ax = plt.subplots(len(IMAGES), 3, figsize=(13, 4.2 * len(IMAGES)))
    for i, (name, path) in enumerate(IMAGES):
        t = C.load_color_target(path, WR, white_thr=sc.white_threshold)
        m = C.subject_mask(t, sc.white_threshold)
        t0 = time.time()
        masks = sam_masks(t)
        lab = masks_to_labels(masks, m)
        print(f"{name:16s} SAM proposed {len(masks):3d} masks -> {int(lab.max()):3d} objects "
              f"({time.time() - t0:.0f}s)")
        old = OS.segment_objects_edges(t, m, k=22, target=18)
        ax[i, 0].imshow(np.clip(t, 0, 1), origin="lower"); ax[i, 0].set_ylabel(name, fontsize=11)
        ax[i, 1].imshow(OS.overlay(t, old), origin="lower")
        ax[i, 2].imshow(OS.overlay(t, lab), origin="lower")
        if i == 0:
            ax[i, 0].set_title("source"); ax[i, 1].set_title("classical edge-merge")
            ax[i, 2].set_title("SAM instances")
        ax[i, 2].set_xlabel(f"{int(lab.max())} objects", fontsize=9)
        for c in range(3): ax[i, c].set_xticks([]); ax[i, c].set_yticks([])
        np.save(f"out_thickness_test/mona_pairs/sam_{name.replace(' ', '_')}.npy", lab)
    plt.suptitle(f"SAM instance segmentation ({MODEL.split('/')[-1]}, {PPS}x{PPS} prompts)",
                 fontsize=14, y=0.999)
    plt.tight_layout()
    plt.savefig("out_thickness_test/mona_pairs/sam_check.png", dpi=100, bbox_inches="tight")
    print("\nsaved out_thickness_test/mona_pairs/sam_check.png (+ per-image .npy label maps)")
