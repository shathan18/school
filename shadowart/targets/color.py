"""Colour: perspex palette, CMYK separation, colour-target loading, shard colour quantise.

Physical model (per-fragment tint): each shard is cut from ONE stock colour of transparent
perspex. A coloured transparent shard filters white light by its *transmittance* — cyan
passes green+blue (blocks red) so its shadow looks cyan, etc. We snap each shard's target
colour (sampled from the source image) to the nearest palette entry by transmittance.

Because the CMYK source art is built from flat primaries, one solid-colour shard per wall
pixel reproduces the picture directly (no subtractive layer-mixing needed).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# name -> (display RGB for viewing/cut files, transmittance RGB of ONE sheet of that perspex).
# Transmittances are *partial* (real coloured acrylic, not ideal filters) so that STACKING
# two sheets of a colour darkens it (T^2) and mixing different colours reads realistically.
PERSPEX = OrderedDict([
    ("clear", ((1.00, 1.00, 1.00), (1.00, 1.00, 1.00))),   # background: no shard
    # transmittance = ONE sheet; tuned light so ~2 stacked sheets ≈ the full saturated colour
    # (1 layer = pale tone, 2 layers = full). Keeps stacking-as-tone without over-darkening.
    ("C",     ((0.00, 0.68, 0.86), (0.35, 0.90, 0.96))),   # cyan: absorbs red
    ("M",     ((0.85, 0.00, 0.50), (0.92, 0.35, 0.86))),   # magenta: absorbs green
    ("Y",     ((1.00, 0.88, 0.00), (0.97, 0.92, 0.32))),   # yellow: absorbs blue
    ("K",     ((0.10, 0.10, 0.12), (0.30, 0.30, 0.32))),   # black: absorbs all
    # secondary colours kept for reference only (not used by CMYK colour mode)
    ("R",     ((0.86, 0.15, 0.15), (0.90, 0.30, 0.30))),
    ("G",     ((0.15, 0.65, 0.28), (0.30, 0.80, 0.35))),
    ("B",     ((0.15, 0.25, 0.80), (0.30, 0.34, 0.88))),
    # --- NOIR palette: neutral grey ramp (equal RGB channels, descending brightness) so
    # the projection reads as tonal black-and-white; tone still comes from stacking sheets.
    ("GRAY_L", ((0.72, 0.72, 0.72), (0.78, 0.78, 0.78))),   # light grey (pale single sheet)
    ("GRAY_M", ((0.48, 0.48, 0.48), (0.58, 0.58, 0.58))),   # mid grey
    ("GRAY_D", ((0.26, 0.26, 0.26), (0.42, 0.42, 0.42))),   # dark grey (K handles pure black)
    # --- MUTED palette: desaturated C/M/Y (transmittances pulled toward neutral so the low
    # channels sit higher and the high-channel spread is compressed -> dusty, low-chroma).
    # Kept PARTIAL like the CMYK set, so ~2 stacked sheets still approach the full muted tone.
    ("C_MUTE", ((0.35, 0.60, 0.66), (0.60, 0.84, 0.88))),   # dusty cyan
    ("M_MUTE", ((0.66, 0.40, 0.55), (0.86, 0.60, 0.82))),   # dusty magenta/rose
    ("Y_MUTE", ((0.72, 0.68, 0.42), (0.90, 0.86, 0.60))),   # dusty ochre/yellow
])
CUT_COLORS = [n for n in PERSPEX if n != "clear"]        # colours that become physical shards
CMYK = ["C", "M", "Y", "K"]                              # process channels used by colour mode

# Named palette presets, selectable via scene `color.preset` or CLI `--palette`. Each is an
# ordered list of PERSPEX keys (excluding 'clear', which palette_names re-adds). 'noir' has no
# C/M/Y members, so decompose falls back to direct nearest-colour quantisation for it (see
# fragment_shards_overlap); 'cmyk'/'muted' keep the CMYK separation path.
PALETTES = {
    "cmyk":  ["C", "M", "Y", "K"],
    "muted": ["C_MUTE", "M_MUTE", "Y_MUTE", "K"],
    "noir":  ["GRAY_L", "GRAY_M", "GRAY_D", "K"],
}


def has_cmyk_channels(names):
    """True if the palette uses the C/M/Y process channels (so the CMYK-separation shard path
    applies). A palette with none of C/M/Y (e.g. 'noir') uses direct nearest-colour quantise."""
    return any(n in ("C", "M", "Y") for n in names)


def display_rgb(name):
    return np.array(PERSPEX[name][0], dtype=np.float32)


def transmit_rgb(name):
    return np.array(PERSPEX[name][1], dtype=np.float32)


def hex_color(name):
    r, g, b = (int(round(c * 255)) for c in PERSPEX[name][0])
    return f"#{r:02x}{g:02x}{b:02x}"


def palette_names(subset=None):
    """Ordered palette names (incl. 'clear'), optionally restricted to `subset`."""
    if not subset:
        return list(PERSPEX)
    keep = {"clear", *subset}
    return [n for n in PERSPEX if n in keep]


def transmit_lut(names):
    """[len(names),3] transmittance table so panel_T = lut[colorid]."""
    return np.array([PERSPEX[n][1] for n in names], dtype=np.float32)


def stack_transmit_lut(names, stack_colorid, stack_intensity=None):
    """Combine a laminated stack of colour-ids into one per-panel-pixel transmittance.

    stack_colorid: [S,P,Hp,Wp] int (0 = clear at that stack slot). Physically each slot is a
    thin sub-layer at the same (u,v) footprint but a different micro depth-offset, so light
    passes through all of them in series -> transmittances multiply (subtractive mixing).
    Returns [P,Hp,Wp,3] to feed straight into Renderer.render_color_np.

    `stack_intensity` [S,P,Hp,Wp] float in [0,1] (optional): the CMYK intensity of each
    slot's channel in the source pixel. When given, each sheet is applied INTENSITY-WEIGHTED,
    `T *= 1 - intensity*(1 - sheet)`, instead of at full strength. This is the hue fix: the
    old full-strength binary lamination discarded channel intensity, so golden-orange
    (strong Y, weak M) laminated identically to pure red (strong Y, strong M) -- both
    "M and Y on" -> the same maroon. Weighting the weak M by its 0.28 intensity keeps the
    blue/green channels high and preserves the golden hue. intensity=1 recovers the old
    behaviour exactly (`1 - 1*(1-sheet) = sheet`); intensity=0 is no tint; clear slots
    (colour-id 0, sheet=(1,1,1)) stay neutral for any intensity.

    NOTE (kept deliberately separate, still unfixed): this is a *render-side* density
    approximation -- the physical cut piece is still one full sheet of its channel
    (`panel_stack_pieces`/cut files unchanged). True physical tone/saturation would come
    from STACKING several same-channel sheets, which overlap mode still doesn't do (it lays
    one sheet per distinct channel) -- that's the separate desaturation issue, not this fix.
    """
    lut = transmit_lut(names)                                # [K,3]
    T = np.ones(stack_colorid.shape[1:] + (3,), dtype=np.float32)
    for s in range(stack_colorid.shape[0]):
        sheet = lut[stack_colorid[s]]                        # [P,Hp,Wp,3]
        if stack_intensity is None:
            T *= sheet
        else:
            w = stack_intensity[s][..., None]                # [P,Hp,Wp,1]
            T *= 1.0 - w * (1.0 - sheet)
    return T


def display_lut(names):
    return np.array([PERSPEX[n][0] for n in names], dtype=np.float32)


def quantize(rgb, names=None):
    """Nearest palette entry (by transmittance colour) for an RGB triple in [0,1]."""
    names = names or list(PERSPEX)
    rgb = np.asarray(rgb, dtype=np.float32)
    best, bd = "clear", 1e9
    for n in names:
        d = float(np.sum((rgb - transmit_rgb(n)) ** 2))
        if d < bd:
            bd, best = d, n
    return best


def quantize_ids(px, names):
    """Vectorised nearest-palette id for many pixels. px [N,3] -> ids [N] into `names`."""
    lut = transmit_lut(names)                             # [K,3]
    d = ((np.asarray(px, np.float32)[:, None, :] - lut[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def majority_color(px, names):
    """Most common palette colour among pixels px [N,3] (per-pixel quantise then vote)."""
    ids = quantize_ids(px, names)
    if ids.size == 0:
        return "clear"
    return names[int(np.bincount(ids, minlength=len(names)).argmax())]


# ---------------------------------------------------------------------------
# colour target loading (fit source image to the wall, colour-preserving)
# ---------------------------------------------------------------------------
def load_color_target(path, wall_res, content_frac=0.92, white_thr=0.90):
    """Load an RGB image, crop to its non-white subject, fit centred on a white square wall
    canvas, and return an RGB float map [Hn, Wn, 3] oriented so image-top -> wall-top."""
    Hn, Wn = wall_res
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    sub = subject_mask(arr, white_thr)
    sub = ndimage.binary_opening(sub, iterations=2)       # drop speckle/corner noise
    lbl, n = ndimage.label(sub)
    if n:                                                 # crop to the largest blob (robust)
        counts = np.bincount(lbl.ravel()); counts[0] = 0
        ys, xs = np.where(lbl == counts.argmax())
        img = img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    cw, ch = img.size
    scale = content_frac * min(Wn / cw, Hn / ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (Wn, Hn), (255, 255, 255))
    canvas.paste(img, ((Wn - nw) // 2, (Hn - nh) // 2))
    out = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.flipud(out).copy()                          # image-top -> wall-top (z up)


def subject_mask(rgb, white_thr=0.90, sat_thr=0.12):
    """Boolean mask of the coloured/dark subject (robust to JPEG noise near white).

    A pixel is subject if it is clearly colourful (saturation above `sat_thr`) OR dark
    (brightness at or below `white_thr`). Low-saturation bright pixels (white/near-white,
    incl. compression speckle) are excluded so the crop stays tight."""
    rgb = np.asarray(rgb, dtype=np.float32)
    mx = rgb.max(axis=-1); mn = rgb.min(axis=-1)
    return ((mx - mn) >= sat_thr) | (mx <= white_thr)


# ---------------------------------------------------------------------------
# CMYK separation (for the channel-preview images)
# ---------------------------------------------------------------------------
def rgb_to_cmyk(rgb):
    """One RGB triple in [0,1] -> {'C','M','Y','K'} in [0,1]."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    k = 1.0 - max(r, g, b)
    d = max(1e-6, 1.0 - k)
    clip = lambda v: min(1.0, max(0.0, v))
    return {"C": clip((1 - r - k) / d), "M": clip((1 - g - k) / d),
            "Y": clip((1 - b - k) / d), "K": clip(k)}


def dominant_rgb(px):
    """Most common colour among pixels px [N,3] (mode over a coarse quantisation).

    Robust to a shard straddling a colour boundary: returns the majority colour rather than a
    blended average (which would fabricate false mixes)."""
    px = np.asarray(px, np.float32).reshape(-1, 3)
    if px.shape[0] == 0:
        return np.ones(3, np.float32)
    q = np.round(px * 5) / 5.0                            # 0.2 bins
    uniq, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    top = int(cnt.argmax())
    return px[inv == top].mean(axis=0)                    # refine within the dominant bin


def naive_cmyk(rgb):
    """RGB [H,W,3] in [0,1] -> {'C','M','Y','K'} coverage maps in [0,1]."""
    rgb = np.clip(np.asarray(rgb, float), 0, 1)
    k = 1.0 - rgb.max(axis=-1)
    denom = np.clip(1.0 - k, 1e-6, 1.0)
    c = (1.0 - rgb[..., 0] - k) / denom
    m = (1.0 - rgb[..., 1] - k) / denom
    y = (1.0 - rgb[..., 2] - k) / denom
    return {"C": np.clip(c, 0, 1), "M": np.clip(m, 0, 1),
            "Y": np.clip(y, 0, 1), "K": np.clip(k, 0, 1)}


def save_cmyk_channels(rgb, out_dir, stem):
    """Save the 4 CMYK separation maps of `rgb` as grayscale PNGs (255 = full channel)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ch = naive_cmyk(rgb)
    paths = []
    for name, m in ch.items():
        img = (np.flipud(np.clip(m, 0, 1)) * 255).astype(np.uint8)
        p = out_dir / f"{stem}_{name}.png"
        Image.fromarray(img).save(p)
        paths.append(str(p))
    return paths
