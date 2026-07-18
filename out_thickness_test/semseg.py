"""
SEMANTIC segmentation with pretrained models -- "this is hair, this is the vase, this is sky".

Why this exists: the classical segmenters in objectseg.py cannot do this. Colour clustering
fractures a shaded vase and merges touching flowers; edge-merging fixes coherence but still
cannot tell hair from face, because there is no strong edge between them. Naming a region needs
a model that knows what hair is.

Two small SegFormer models (CPU, no torchvision needed):
  * jonathandinu/face-parsing            -> skin, hair, cloth, neck, nose, eyes, brows, ears, lips
  * nvidia/segformer-b0-finetuned-ade-512-512 -> ADE20K's 150 scene classes (flower, plant, vase,
    tree, sky, water, house, person, ...)

Semantic models label CLASSES, not instances, so "each flower / each star is its own object"
comes from a hybrid: take the model's class region, then split it into instances with
objectseg.split_touching. Model supplies meaning, the classical step supplies separation.

Everything degrades gracefully: if a model returns nothing usable on a stylised painting, the
caller falls back to objectseg.segment_objects_edges rather than shipping a worse division.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

import objectseg as OS

FACE_MODEL = "jonathandinu/face-parsing"
SCENE_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
_CACHE = {}

# CelebAMask-HQ label order used by the face-parsing checkpoint
FACE_LABELS = ["background", "skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow",
               "l_ear", "r_ear", "mouth", "u_lip", "l_lip", "hair", "hat", "ear_r",
               "neck_l", "neck", "cloth"]
# ADE20K classes worth splitting into individual objects (there are usually several of them)
MULTI_CLASSES = {"flower", "plant", "tree", "light", "lamp", "sconce", "chandelier",
                 "pot", "vase", "fruit", "food", "person", "boat", "house", "building"}


def _load(kind):
    if kind in _CACHE:
        return _CACHE[kind]
    from transformers import (AutoImageProcessor, AutoModelForSemanticSegmentation,
                              SegformerImageProcessor)
    name = FACE_MODEL if kind == "face" else SCENE_MODEL
    try:
        proc = AutoImageProcessor.from_pretrained(name)
    except (ValueError, OSError):
        # some older checkpoints (e.g. the face-parsing one) ship a preprocessor_config.json
        # that AutoImageProcessor can no longer identify -- both models ARE SegFormer, so build
        # its processor directly with the standard ImageNet normalisation it was trained with.
        proc = SegformerImageProcessor(do_resize=True, size={"height": 512, "width": 512},
                                       do_normalize=True,
                                       image_mean=[0.485, 0.456, 0.406],
                                       image_std=[0.229, 0.224, 0.225])
    model = AutoModelForSemanticSegmentation.from_pretrained(name).eval()
    _CACHE[kind] = (proc, model)
    return _CACHE[kind]


def _predict(rgb, kind):
    """Class-id map at the input resolution. `rgb` float [H,W,3] in [0,1] (wall orientation)."""
    import torch
    proc, model = _load(kind)
    H, W = rgb.shape[:2]
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits                       # [1,C,h,w]
    up = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    return up.argmax(1)[0].cpu().numpy().astype(int)


def class_names(kind):
    if kind == "face":
        return {i: n for i, n in enumerate(FACE_LABELS)}
    _proc, model = _load("scene")
    return {int(k): v for k, v in model.config.id2label.items()}


def to_regions(rgb, mask, kind="scene", split_multi=True, min_frac=0.002, fallback=True):
    """Model classes -> ShadowArt region ids (1..N, 0 = background).

    Returns (labels, info) where info maps region id -> the class name it came from, so callers
    can give e.g. `skin` a finer shard spacing than `cloth`. Falls back to the classical
    edge-merged segmentation when the model finds too little inside the subject (the
    painting/photo domain gap)."""
    cls = _predict(rgb, kind)
    names = class_names(kind)
    out = np.zeros(rgb.shape[:2], int)
    info, nxt = {}, 0
    min_px = max(30, int(min_frac * max(int(mask.sum()), 1)))
    for c in np.unique(cls):
        name = names.get(int(c), str(c))
        if kind == "face" and name == "background":
            continue
        region = mask & (cls == c)
        if region.sum() < min_px:
            continue
        pieces = (OS.split_touching(region)
                  if (split_multi and name in MULTI_CLASSES) else region.astype(int))
        for p in range(1, int(pieces.max()) + 1):
            piece = pieces == p
            if piece.sum() < min_px:
                continue
            nxt += 1
            out[piece] = nxt
            info[nxt] = name
    covered = float((out > 0).sum()) / max(int(mask.sum()), 1)
    if nxt == 0 or covered < 0.35:                            # model found ~nothing usable here
        if fallback:
            lab = OS.segment_objects_edges(rgb, mask, k=22, target=18)
            return lab, {v: "classical" for v in range(1, int(lab.max()) + 1)}
        return out, info
    holes = mask & (out == 0)                                 # grow objects over the leftovers
    if holes.any():
        inds = ndimage.distance_transform_edt(out == 0, return_indices=True, return_distances=False)
        out = np.where(holes, out[tuple(inds)], out)
    out[~mask] = 0
    return out, info


# finer shards on the parts that carry identity, coarser on flat clothing
FACE_SCALE = {"skin": 0.55, "nose": 0.45, "l_eye": 0.4, "r_eye": 0.4, "l_brow": 0.45,
              "r_brow": 0.45, "mouth": 0.45, "u_lip": 0.45, "l_lip": 0.45,
              "l_ear": 0.7, "r_ear": 0.7, "hair": 0.85, "neck": 1.0, "cloth": 1.2}


def part_scales(info, table=None):
    """{region id: spacing multiplier} from each region's class name (<1 = denser shards)."""
    table = FACE_SCALE if table is None else table
    return {rid: table[name] for rid, name in info.items() if name in table}


def summarise(info):
    from collections import Counter
    return Counter(info.values())


def importance_map(labels, info=None, mode="boundary", width=3, parts=("skin", "l_eye", "r_eye",
                                                                       "nose", "mouth")):
    """A [0,1] map for the SOFT use of segmentation (`semantic_masks` + `semantic_weight`).

    The HARD path (`region_masks`) forbids shards from crossing an object boundary, which costs
    extra shards (one per region, plus border slivers). The SOFT path instead just tells
    `_importance_map` where detail matters, so smaller shards concentrate there while the tiling
    stays free -- no shard inflation. Modes:
      boundary : object EDGES weighted up -- sharpen outlines without partitioning
      parts    : identity-carrying parts (skin/eyes/nose/mouth) weighted up
      both     : union of the two
    """
    labels = np.asarray(labels)
    out = np.zeros(labels.shape, float)
    if mode in ("boundary", "both"):
        edge = np.zeros(labels.shape, bool)
        edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
        edge &= labels > 0
        out = np.maximum(out, ndimage.binary_dilation(edge, iterations=width).astype(float))
    if mode in ("parts", "both") and info:
        want = np.isin(labels, [rid for rid, n in info.items() if n in parts])
        out = np.maximum(out, want.astype(float))
    return out
