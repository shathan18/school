"""Fabrication-constrained optimisation of the CS x Technion piece.

THE POINT OF THIS FILE
----------------------
Every score this repo has ever reported was computed from the RASTER opacity field. That
is not what gets built. What gets built is the output of `panel_stack_pieces`, which runs
`enforce_min_feature` (morphological open+close with a disk of diameter `fab.min_feature`)
and then a kerf buffer. At the old 4 mm feature floor the two agree to within noise, so it
never mattered. At the 2 cm floor this brief requires, they will not agree, and optimising
the raster score would tune a sculpture that cannot be manufactured.

There is a second, independent reason the raster score flatters itself, and it is written
down in the library already -- `stack_transmit_lut` says of its intensity weighting:

    "this is a *render-side* density approximation -- the physical cut piece is still one
     full sheet of its channel (`panel_stack_pieces`/cut files unchanged)"

So the renderer is allowed to use partial-strength ink that the fabricator cannot produce.
With a single hand-mixed tone there is no such thing as a half-strength shard.

`fab_render` therefore closes both gaps: vectorise exactly as the exporter does, rasterise
the resulting polygons back onto the panel grid, and re-render at FULL ink strength. Every
configuration in this study reports both numbers and the delta between them.

LOCKED CONSTRAINTS (confirmed with the user)
--------------------------------------------
  body            0.60 x 0.60 m, walls stay 1.80 m  -> scenes/logo60.yaml
  min shard       2 x 2 cm on the PHYSICAL PERSPEX  -> fab.min_feature = 0.020 (panel space)
  planes          >=1 usable physical crossing AND >=2 planes serving both walls
  palette         ONE hand-mixed alcohol ink + clear Perspex
  illumination    installation never moves; only which lamp is lit changes

Run:
  python logo_opt.py --baseline     Phase 1: port the shipped piece and measure the honest cost
  python logo_opt.py --selftest     Phase 0: prove the fab harness agrees with the exporter
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import shapely
from PIL import Image

from shadowart import metrics as _metrics
from shadowart.config.io import load_scene
from shadowart.geometry.panels import weave_crossings, slot_depth
from shadowart.solve import decompose
from shadowart.targets import color as C

import face_pretest as FP
import face_paper_floor as PF
import face_duty_sweep as DS
import logo_pretest as LP
import logo_render as LR
import logo_final as LF

SCENE = "scenes/logo60.yaml"
OUT = Path("out_logo_opt")

PAIR = ("cs", "technion5")     # locked subject
MARGIN = LF.MARGIN             # 0.12 white border around the mark

MIN_SHARD = 0.020              # metres, PANEL space -- the 2 cm hand-colouring floor
BODY = 0.60                    # metres, the installation envelope
BODY_X = (1.70, 2.30)          # the 60 cm footprint, matching scenes/logo60.yaml
BODY_Y = (1.70, 2.30)
BODY_Z = (0.78, 1.38)
MIN_PANEL_U = 0.12             # a panel clipped below this is not worth cutting

# A crossing is only buildable if the slot is not absurdly wide. slot_width =
# thickness / sin(angle), so 3 mm at 90 deg, 4.2 mm at 45 deg, 6 mm at 30 deg, 17 mm at 10.
MIN_CROSS_ANGLE = 30.0         # degrees
MIN_CROSS_Z = 0.05             # metres of shared height -- a 5 cm tab is the least worth cutting

# FACE-ON IS 135 deg IN THIS SCENE, NOT 45. Measured, not assumed: given a free choice
# between a 43-47 and a 133-137 band, the coverage-greedy search picked 133-137 for all ten
# panels. With the lamps at (3.00, 2.55) and (2.55, 3.00), a 45 deg panel lies almost ALONG
# both light rays -- seen edge-on, it casts a narrow shadow and the search correctly refuses
# it. (The old room-scale scene had lamps at (3.00, 0.70) and (0.70, 3.00), where 45 deg IS
# face-on; the 43-47 band was inherited from there and is simply wrong here.)
#
# So the two bands must straddle 135 SYMMETRICALLY, or the search collapses onto whichever
# band is nearer face-on and no crossing can ever form. Separation also has to clear
# MIN_CROSS_ANGLE: slot_width = thickness / sin(separation).
BANDS_SINGLE = None                        # reproduces the shipped layout
BANDS_CROSS = [(105, 120), (150, 165)]     # symmetric about 135; separation 30-60 deg

# Panel-orientation sweep. Wider separation buys a tighter slot but costs shadow width,
# since deviation from face-on narrows the cast shadow by cos(deviation).
BAND_SETS = {
    "sep20_40": [(115, 125), (145, 155)],   # cos dev 0.97-0.99, slot 4.2-8.8 mm
    "sep30_60": [(105, 120), (150, 165)],   # cos dev 0.87-0.97, slot 3.5-6.0 mm
    "sep60_90": [(90, 105), (165, 180)],    # cos dev 0.71-0.87, slot 3.0-3.5 mm
    "wide": [(100, 130), (140, 170)],       # loose, lets the search choose
}

INK = "INK"                    # runtime-registered single hand-mixed tone

# The panel-search box MUST match the 60 cm body or the greedy search places panels outside
# the installation. Read once from the scene so the yaml stays the single source of truth.
_S0 = load_scene(SCENE)
PANEL_KW = dict(
    u_size_range=tuple(_S0.solve.search_u_size_range),
    v_range=tuple(_S0.solve.search_v_range),
    anchor_range=tuple(_S0.solve.search_anchor_range),
    standoff=float(_S0.solve.search_standoff),
    mag_cap=float(_S0.solve.search_mag_cap),
)


# --------------------------------------------------------------------------------------
# ink + scene
# --------------------------------------------------------------------------------------
def install_ink(t: float) -> str:
    """Register the hand-mixed tone as a palette entry.

    The stock presets (GRAY_L/M/D) are fixed transmittances for pre-tinted acrylic. This
    piece is clear Perspex coloured by hand, so the tone is a FREE CONTINUOUS parameter --
    a lever no preset exposes. Neutral, so display == transmittance.
    """
    C.PERSPEX[INK] = ((t, t, t), (t, t, t))
    if INK not in C.CUT_COLORS:
        C.CUT_COLORS.append(INK)
    return INK


def make_scene_fn(min_feature=MIN_SHARD, lights=None, solve_kw=None, scene_kw=None):
    """Scene mutation applied inside `build_floor`, after its noir default and before density.

    Two dicts, because the knobs live at two levels: `fragment_size` and the search ranges are
    fields of `scene.solve`, while `overlap_shard_budget`, `overlap_detail_bias` and
    `source_radius` are fields of `Scene` itself.
    """
    def fn(scene):
        scene = dataclasses.replace(
            scene,
            color_palette=[INK],          # one ink; `clear` is re-added by palette_names
            color_max_stack=1,            # a single tone cannot laminate to a second one
            color_max_layers=1,
            fab=dataclasses.replace(scene.fab, min_feature=min_feature),
        )
        if scene_kw:
            scene = dataclasses.replace(scene, **scene_kw)
        if solve_kw:
            scene = dataclasses.replace(scene, solve=dataclasses.replace(scene.solve, **solve_kw))
        if lights:
            scene = dataclasses.replace(scene, lights={
                k: dataclasses.replace(v, pos=tuple(lights[k])) if k in lights else v
                for k, v in scene.lights.items()})
        return scene
    return fn


def write_targets(keys, tone: float) -> dict:
    """`greymark_m` tonal design: mark at ONE ink tone on a bare-wall (clear Perspex) ground.

    The white ground sits above `white_threshold`, so it is not subject, never scores, and
    needs no material -- which is exactly what kept this design legible at IoU 0.94 while
    every black-grounded arm was capped by the palette rather than by the layout.
    """
    PF.TGT.mkdir(parents=True, exist_ok=True)
    names = {}
    cands = {}
    for key in keys:
        two = FP.posterize_gray(LP.load_square(key, MARGIN), 2)
        img = np.where(two > 0.5, 1.0, tone)
        name = f"{key}_ink{int(round(tone * 100)):03d}"
        Image.fromarray((np.clip(np.stack([img] * 3, -1), 0, 1) * 255).astype(np.uint8)).save(
            PF.TGT / f"{name}_f100.png")
        names[key] = name
        cands[name] = LR.Shim()
    return names, cands


def clip_to_body(panels, min_u=MIN_PANEL_U):
    """Trim every searched panel to the 60 cm body.

    `search_anchor_range` constrains panel CENTRES, not extents, so a centred panel of
    u_size 0.60 overshoots the body by up to 0.21 m on each axis -- measured at Phase 1 as a
    0.68 x 0.72 m envelope against a 0.60 m spec. Clipping the u_range (rather than shrinking
    the search box) is what a fabricator would actually do, and it preserves the depth
    diversity that off-centre panels provide: a panel is kept where it was, just cut shorter.

    Panels clipped below `min_u` are dropped, so the caller must ask for more panels than it
    needs and check how many survived.
    """
    out = []
    for p in panels:
        dx, dy = float(np.cos(p.angle)), float(np.sin(p.angle))
        lo, hi = float(p.u_range[0]), float(p.u_range[1])
        ok = True
        for a, d, (m0, m1) in ((p.anchor[0], dx, BODY_X), (p.anchor[1], dy, BODY_Y)):
            if abs(d) < 1e-9:                       # parallel to this axis: no u constraint
                if not (m0 - 1e-9 <= a <= m1 + 1e-9):
                    ok = False                      # ...but the whole panel is outside
                continue
            t0, t1 = (m0 - a) / d, (m1 - a) / d
            lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
        if not ok or hi - lo < min_u:
            continue
        v0 = max(float(p.v_range[0]), BODY_Z[0])
        v1 = min(float(p.v_range[1]), BODY_Z[1])
        if v1 - v0 < min_u:
            continue
        out.append(dataclasses.replace(p, u_range=(lo, hi), v_range=(v0, v1)))
    return out


def usable_crossings(layout):
    """Crossings a laser and a pair of hands can actually make.

    slot_width = thickness / sin(angle), so a near-parallel crossing needs an absurd notch:
    3 mm at 90 deg, 6 mm at 30 deg, 17 mm at 10 deg. Phase 1's single 43-47 deg band produced
    exactly one crossing, at 4 deg -- a 43 mm slot, i.e. no joint at all.
    """
    return [c for c in weave_crossings(layout)
            if c.angle_deg >= MIN_CROSS_ANGLE and (c.z_range[1] - c.z_range[0]) >= MIN_CROSS_Z]


def build(ka, kb, *, seed, panel_count, bands, density, ink_t, tone=None,
          min_feature=MIN_SHARD, lights=None, solve_kw=None, scene_kw=None, clip=True):
    """One candidate installation. Returns the standard `build_floor` bundle."""
    tone = ink_t if tone is None else tone
    install_ink(ink_t)
    names, cands = write_targets((ka, kb), tone)
    pk = dict(PANEL_KW)
    if bands:
        pk["angle_bands"] = list(bands)
    return PF.build_floor(
        ("logo", names[ka], names[kb]), seed, 1.0, "uniform", cands,
        panel_count=panel_count, angle_range=(43, 47), density=density,
        target_kw=dict(crop_mode="all", content_frac=0.88),
        scene_path=SCENE, scene_fn=make_scene_fn(min_feature, lights, solve_kw, scene_kw),
        panel_kw=pk, panel_post=clip_to_body if clip else None)


# --------------------------------------------------------------------------------------
# PHASE 0 -- fabrication-faithful evaluation
# --------------------------------------------------------------------------------------
def _rasterise(poly, u_range, v_range, Hp, Wp):
    """Inverse of `contours.mask_to_polygons`: u = u0 + col*su, v = v0 + row*sv.

    A pixel is ON iff its CENTRE lies inside the polygon. That is the exact inverse of
    marching squares, which traces the 0.5 level line running BETWEEN an on-centre and an
    off-centre. Filling the same contour with a scanline rasteriser instead (PIL, OpenCV)
    erodes every piece by ~half a pixel all the way round -- measured at 8.5% of the mask,
    which would have been silently charged to the 2 cm constraint.

    Holes are handled by `contains_xy` itself, so no separate punch-out pass is needed and
    an island nested inside a hole survives correctly.
    """
    su = (u_range[1] - u_range[0]) / Wp
    sv = (v_range[1] - v_range[0]) / Hp
    minu, minv, maxu, maxv = poly.bounds
    c0 = max(0, int(np.floor((minu - u_range[0]) / su)))
    c1 = min(Wp, int(np.ceil((maxu - u_range[0]) / su)) + 1)
    r0 = max(0, int(np.floor((minv - v_range[0]) / sv)))
    r1 = min(Hp, int(np.ceil((maxv - v_range[0]) / sv)) + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    X, Y = np.meshgrid(u_range[0] + np.arange(c0, c1) * su,
                       v_range[0] + np.arange(r0, r1) * sv)
    return slice(r0, r1), slice(c0, c1), shapely.contains_xy(poly, X, Y)


def enforce_shard_floor(pieces, min_shard=MIN_SHARD):
    """Drop every polygon that cannot contain a `min_shard` disk. Returns (kept, n_dropped).

    This is needed because `enforce_min_feature` assumes ISOTROPIC pixels. The searched
    panels have variable width (u_size 0.20-0.60 m) but fixed height (0.60 m) at a fixed
    420x420 panel_res, so a narrow panel has su = 0.48 mm/px against sv = 1.43 mm/px. The
    disk structuring element is built in PIXELS, so in METRES it is an ellipse, and the
    guarantee degrades to ~10 mm along u. Phase 1 measured shards down to 0.6 mm surviving a
    nominal 20 mm floor.

    Filtering after vectorisation makes the rule true by construction and keeps the fix out
    of the shared library, where it would silently change every other pipeline. Dropping
    (rather than merging into a neighbour) is the conservative reading: it costs fidelity in
    the report instead of hiding the problem.
    """
    kept, dropped = {}, 0
    for name, items in pieces.items():
        keep = []
        for poly, ch, slot in items:
            if inscribed_diameter(poly) >= min_shard - 1e-6:
                keep.append((poly, ch, slot))
            else:
                dropped += 1
        kept[name] = keep
    return kept, dropped


def fab_geometry(b, kerf=None, min_shard=MIN_SHARD):
    """Vectorise exactly as the exporter does, enforce the shard floor, then rasterise back.

    `panel_stack_pieces` is the SAME call `export_fab` makes, so what is scored here and
    what lands in cut/ are one object, not two similar ones.
    """
    layout, names = b["layout"], b["names"]
    sc = b["stack_colorid"]
    pieces = decompose.panel_stack_pieces(layout, sc, names, kerf=kerf)
    pieces, dropped = enforce_shard_floor(pieces, min_shard)
    S, _P, Hp, Wp = sc.shape
    cid_of = {n: i for i, n in enumerate(names)}
    out = np.zeros_like(sc)
    for gi, panel in enumerate(layout.panels):
        for poly, ch, slot in pieces.get(panel.name, []):
            r = _rasterise(poly, panel.u_range, panel.v_range, Hp, Wp)
            if r is None:
                continue
            rs, cs, m = r
            sub = out[slot, gi]
            sub[rs, cs][m] = cid_of[ch]
            out[slot, gi] = sub
    return pieces, out, dropped


def fab_render(b, kerf=None, min_shard=MIN_SHARD):
    """Re-render from the cut geometry at FULL ink strength.

    stack_intensity is deliberately NOT passed: it is a render-side density approximation
    (see the note in `stack_transmit_lut`) and a single hand-mixed tone has no half-strength
    variant. The fabricated shard is one full sheet of ink or it is absent.
    """
    pieces, fcid, dropped = fab_geometry(b, kerf=kerf, min_shard=min_shard)
    panel_T = C.stack_transmit_lut(b["names"], fcid, None)
    rgb = b["renderer"].render_color_np(panel_T)
    return pieces, fcid, panel_T, rgb, dropped


def inscribed_diameter(poly, tol=2e-4) -> float:
    """Diameter of the largest disk that fits inside `poly`.

    This is the correct test for the 2 cm rule because it is the same predicate
    `enforce_min_feature` enforces (opening by a disk). An area or bounding-box test would
    pass a long thin sliver that no one can pick up and hand-colour.
    """
    if poly.is_empty:
        return 0.0
    if hasattr(shapely, "maximum_inscribed_circle"):
        return 2.0 * float(shapely.maximum_inscribed_circle(poly, tolerance=tol).length)
    lo, hi = 0.0, 0.01                                   # bisection fallback, shapely < 2.1
    while not poly.buffer(-hi).is_empty:
        lo = hi
        hi *= 2.0
        if hi > 1.0:
            break
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if poly.buffer(-mid).is_empty:
            hi = mid
        else:
            lo = mid
    return 2.0 * lo


def fab_report(b, pieces, dropped=0, min_shard=MIN_SHARD) -> dict:
    """Buildability + fabrication cost. Cost is reported as COMPONENTS, not one opaque number:
    they are paid by different people (laser time vs. hand-colouring hours vs. assembly)."""
    layout = b["layout"]
    polys = [p for items in pieces.values() for (p, _c, _s) in items]
    diam = np.array([inscribed_diameter(p) for p in polys]) if polys else np.zeros(0)
    area = np.array([p.area for p in polys]) if polys else np.zeros(0)
    perim = float(sum(p.length for p in polys))

    xs = weave_crossings(layout)
    usable = usable_crossings(layout)

    # installation envelope: floor-plan extent of every panel, plus height
    pts = np.array([pt for p in layout.panels for pt in p.floor_segment_xy()])
    env_x = float(pts[:, 0].max() - pts[:, 0].min()) if len(pts) else 0.0
    env_y = float(pts[:, 1].max() - pts[:, 1].min()) if len(pts) else 0.0
    env_z = float(max(p.v_range[1] for p in layout.panels)
                  - min(p.v_range[0] for p in layout.panels)) if layout.panels else 0.0

    return dict(
        n_shards=len(polys),
        min_diam_mm=float(diam.min() * 1000) if len(diam) else 0.0,
        mean_diam_mm=float(diam.mean() * 1000) if len(diam) else 0.0,
        min_area_cm2=float(area.min() * 1e4) if len(area) else 0.0,
        mean_area_cm2=float(area.mean() * 1e4) if len(area) else 0.0,
        ink_area_cm2=float(area.sum() * 1e4),
        engrave_m=perim,
        n_undersize=int((diam < min_shard - 1e-6).sum()),
        n_dropped=int(dropped),
        n_panels=len(layout.panels),
        n_crossings=len(xs),
        n_usable_crossings=len(usable),
        max_slot_mm=float(max((c.slot_width for c in usable), default=0.0) * 1000),
        slot_depth_mm=float(slot_depth(layout) * 1000),
        env_x_m=env_x, env_y_m=env_y, env_z_m=env_z,
        fits_body=bool(env_x <= BODY + 1e-6 and env_y <= BODY + 1e-6 and env_z <= BODY + 1e-6),
    )


def mark_iou(target_rgb, wall_rgb) -> float:
    """IoU of the INKED region -- the mark itself.

    NOT `logo_render.mark_iou`, which thresholds at the target's mid-level and takes the IoU
    of the region ABOVE it. On a grey mark over a white ground that selects the BACKGROUND,
    which covers 74-84% of the frame, so it scores high no matter what the mark does:
    measured 0.9239 on a wall whose true mark IoU is 0.6493. Every Phase 1-3 number in this
    file used that metric and is inflated for the same reason.

    The mark is the DARKER region, so that is what gets compared here.
    """
    t = target_rgb.mean(-1)
    r = wall_rgb.mean(-1)
    thr = 0.5 * (float(t.min()) + float(t.max()))
    a, b = t <= thr, r <= thr
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def evaluate(b, label, ka, kb, density, seed) -> dict:
    """Raster score (comparable to every previous number in this repo) + the honest fab score."""
    t0 = time.time()
    base, rows, ras = LF.score(b, label, ka, kb, density, seed=seed)
    pieces, _fcid, _pT, frgb, dropped = fab_render(b)
    facc = _metrics.evaluate_wall_accuracy(b["targets"], frgb)
    fiou = {w: LR.mark_iou(b["targets"][w], frgb[w]) for w in ("A", "B")}
    fmark = {w: mark_iou(b["targets"][w], frgb[w]) for w in ("A", "B")}
    fab = dict(
        ssim=float(np.mean([facc["A"]["ssim"], facc["B"]["ssim"]])),
        edge=float(np.mean([facc["A"]["edge_fidelity"], facc["B"]["edge_fidelity"]])),
        iou_A=fiou["A"], iou_B=fiou["B"], iou=float(np.mean(list(fiou.values()))),
        mark_A=fmark["A"], mark_B=fmark["B"], mark=float(np.mean(list(fmark.values()))),
        mse=float(np.mean([facc["A"]["mse"], facc["B"]["mse"]])),
    )
    fab.update(fab_report(b, pieces, dropped))
    return dict(raster=ras, fab=fab, rows=rows,
                d_ssim=fab["ssim"] - ras["ssim"], d_iou=fab["iou"] - ras["iou"],
                runtime_s=time.time() - t0), base, frgb, pieces


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def show(tag, rec) -> None:
    r, f = rec["raster"], rec["fab"]
    print(f"\n  {tag}")
    print(f"    raster   SSIM {r['ssim']:.4f}  IoU {r['iou']:.4f} (A {r['iou_A']:.4f} B {r['iou_B']:.4f})"
          f"  shards {r['shards']}")
    print(f"    FAB      SSIM {f['ssim']:.4f}  IoU {f['iou']:.4f} (A {f['iou_A']:.4f} B {f['iou_B']:.4f})"
          f"  shards {f['n_shards']}")
    print(f"    MARK IoU {f['mark']:.4f} (A {f['mark_A']:.4f} B {f['mark_B']:.4f})"
          f"   <-- the inked region; the IoU above scores the white ground")
    print(f"    delta    SSIM {rec['d_ssim']:+.4f}  IoU {rec['d_iou']:+.4f}"
          f"   <-- what the raster score was hiding")
    print(f"    planes   {r['n_both']}/{r['n_panels']} serve BOTH walls, {r['n_dead']} dead;"
          f"  neff {r['neff_A']:.2f}/{r['neff_B']:.2f}  busiest carries {100 * r['top']:.0f}%")
    print(f"    shard    min {f['min_diam_mm']:.1f} mm  mean {f['mean_diam_mm']:.1f} mm"
          f"   dropped {f['n_dropped']} undersize"
          f"   {'OK' if f['n_undersize'] == 0 else 'FAIL'}")
    print(f"    weave    {f['n_usable_crossings']} usable of {f['n_crossings']} crossings"
          f"  (max slot {f['max_slot_mm']:.1f} mm, depth {f['slot_depth_mm']:.1f} mm)"
          f"   {'OK' if f['n_usable_crossings'] >= 1 else 'FAIL'}")
    print(f"    envelope {f['env_x_m']:.2f} x {f['env_y_m']:.2f} x {f['env_z_m']:.2f} m"
          f"   {'OK' if f['fits_body'] else 'FAIL'}")
    print(f"    fab cost {f['n_shards']} shards to hand-colour, {f['engrave_m']:.1f} m to engrave,"
          f" {f['n_panels']} panels, {f['n_usable_crossings']} slots,"
          f" {f['ink_area_cm2']:.0f} cm2 ink")
    print(f"    runtime  {rec['runtime_s']:.1f} s")


def save_walls(out, tag, targets, rgb) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for w in ("A", "B"):
        Image.fromarray((np.clip(np.flipud(rgb[w]), 0, 1) * 255).astype(np.uint8)).save(
            out / f"{tag}_wall_{w}.png")
    for w in ("A", "B"):
        Image.fromarray((np.clip(np.flipud(targets[w]), 0, 1) * 255).astype(np.uint8)).save(
            out / f"target_{w}.png")


# --------------------------------------------------------------------------------------
# PHASE 0 self-test
# --------------------------------------------------------------------------------------
def selftest() -> None:
    """Prove the harness before trusting a single number it produces.

    1. round-trip: rasterising the polygons back must reproduce the mask the vectoriser saw,
       otherwise `fab_render` is measuring my rasteriser rather than the sculpture.
    2. at a TINY min_feature the fab score must nearly equal the raster score (nothing is
       removed, so the only difference is quantisation).
    """
    print("PHASE 0 self-test -- validating the fabrication-faithful harness")
    ka, kb = PAIR
    b = build(ka, kb, seed=8, panel_count=10, bands=BANDS_CROSS, density=0.70,
              ink_t=0.58, tone=0.53, min_feature=0.002)

    pieces, fcid, dropped = fab_geometry(b, min_shard=0.0)
    sc = b["stack_colorid"]
    inter = int(((fcid > 0) & (sc > 0)).sum())
    union = int(((fcid > 0) | (sc > 0)).sum())
    print(f"  round-trip IoU(raster mask, re-rasterised polygons) = {inter / max(union, 1):.4f}"
          f"   [{inter} / {union} px]")

    panel_T = C.stack_transmit_lut(b["names"], fcid, None)
    rgb = b["renderer"].render_color_np(panel_T)
    acc = _metrics.evaluate_wall_accuracy(b["targets"], rgb)
    base = b["renderer"].render_color_np(b["panel_T"])
    acc0 = _metrics.evaluate_wall_accuracy(b["targets"], base)
    s1 = float(np.mean([acc["A"]["ssim"], acc["B"]["ssim"]]))
    s0 = float(np.mean([acc0["A"]["ssim"], acc0["B"]["ssim"]]))
    print(f"  at min_feature 2 mm:  raster SSIM {s0:.4f}   fab SSIM {s1:.4f}   delta {s1 - s0:+.4f}")

    polys = [p for items in pieces.values() for (p, _c, _s) in items]
    d = sorted(inscribed_diameter(p) * 1000 for p in polys)
    print(f"  inscribed diameters (mm): min {d[0]:.1f}  p05 {d[len(d) // 20]:.1f}  "
          f"median {d[len(d) // 2]:.1f}  max {d[-1]:.1f}   n={len(d)}")
    print("  -> harness is sound if round-trip IoU > 0.95 and the 2 mm delta is small.")


# --------------------------------------------------------------------------------------
# PHASE 1 baseline
# --------------------------------------------------------------------------------------
def gates(rec) -> dict:
    """The four hard requirements. A configuration that fails any of them is not a design."""
    r, f = rec["raster"], rec["fab"]
    return dict(
        shard_2cm=f["n_undersize"] == 0,
        crossing=f["n_usable_crossings"] >= 1,
        both_walls=r["n_both"] >= LF.MIN_BOTH,
        envelope=f["fits_body"],
    )


def baseline() -> None:
    """Port the shipped piece onto the 60 cm body under the real 2 cm floor.

    Configurations are cumulative so the cost of each constraint is separated, not lumped:
      A  shipped-equivalent : 4 mm floor, single angle band, unclipped   (reference)
      B  + the 2 cm hand-colouring floor
      C  + the two-band layout that is geometrically capable of crossing
      D  + body clipping and the enforced shard floor  (the first buildable candidate)
    """
    OUT.mkdir(parents=True, exist_ok=True)
    ka, kb = PAIR
    cfgs = [
        ("A_4mm_singleband", dict(min_feature=0.004, bands=BANDS_SINGLE, clip=False)),
        ("B_20mm_singleband", dict(min_feature=MIN_SHARD, bands=BANDS_SINGLE, clip=False)),
        ("C_20mm_crossbands", dict(min_feature=MIN_SHARD, bands=BANDS_CROSS, clip=False)),
        ("D_20mm_cross_clip", dict(min_feature=MIN_SHARD, bands=BANDS_CROSS, clip=True)),
    ]
    recs = {}
    for tag, kw in cfgs:
        print(f"\n=== {tag} ===", flush=True)
        b = build(ka, kb, seed=8, panel_count=10, density=0.70, ink_t=0.58, tone=0.53, **kw)
        rec, base, frgb, _pieces = evaluate(b, "logo", ka, kb, 0.70, 8)
        rec["gates"] = gates(rec)
        show(tag, rec)
        save_walls(OUT / tag, "fab", b["targets"], frgb)
        save_walls(OUT / tag, "raster", b["targets"], base)
        recs[tag] = rec
    _summarise(recs, "baseline.json")


def feasible(seeds=range(12), panel_count=12) -> None:
    """Phase 1b: find a seed whose CLIPPED two-band layout actually satisfies all four gates.

    The layout seed is the dominant lever on how work is spread across planes (measured
    earlier: seed 8 gave 8/10 planes at neff 7.2/5.7, seed 7 gave 2/10 at 4.5/1.6), and it
    is also what decides whether any two panels physically cross. Nothing else in the
    pipeline controls either, so this is the right variable to sweep first.

    `panel_count` is raised above the target because `clip_to_body` drops panels whose
    clipped span falls below MIN_PANEL_U.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    ka, kb = PAIR
    recs = {}
    for s in seeds:
        tag = f"seed{s:02d}"
        b = build(ka, kb, seed=s, panel_count=panel_count, density=0.70,
                  ink_t=0.58, tone=0.53, bands=BANDS_CROSS, clip=True)
        rec, _base, frgb, _pieces = evaluate(b, "logo", ka, kb, 0.70, s)
        rec["gates"] = gates(rec)
        g = rec["gates"]
        f, r = rec["fab"], rec["raster"]
        print(f"  {tag}  panels {r['n_panels']:>2d}  both {r['n_both']:>2d}  "
              f"cross {f['n_usable_crossings']:>2d}  shards {f['n_shards']:>3d}  "
              f"fabIoU {f['iou']:.4f}  fabSSIM {f['ssim']:.4f}  "
              f"neff {r['neff_A']:.1f}/{r['neff_B']:.1f}  "
              f"{'PASS' if all(g.values()) else 'fail:' + ','.join(k for k, v in g.items() if not v)}",
              flush=True)
        if all(g.values()):
            save_walls(OUT / tag, "fab", b["targets"], frgb)
        recs[tag] = rec
    ok = {k: v for k, v in recs.items() if all(v["gates"].values())}
    print(f"\n  {len(ok)} of {len(recs)} seeds satisfy all four gates")
    if ok:
        best = max(ok, key=lambda k: ok[k]["fab"]["iou"])
        print(f"  best feasible: {best}  fabIoU {ok[best]['fab']['iou']:.4f}")
        show(best, ok[best])
    _summarise(recs, "feasible.json")


def bandsweep(seeds=range(6)) -> None:
    """Phase 2a: panel orientation. The first axis of the sweep, because it is the one that
    decides whether the piece can be joined at all -- fidelity is moot if it cannot be built.

    Reports, for each band set, how often a seed yields a layout passing all four gates.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    ka, kb = PAIR
    recs = {}
    for bname, bands in BAND_SETS.items():
        for s in seeds:
            tag = f"{bname}_s{s:02d}"
            b = build(ka, kb, seed=s, panel_count=12, density=0.70,
                      ink_t=0.58, tone=0.53, bands=bands, clip=True)
            rec, _base, frgb, _p = evaluate(b, "logo", ka, kb, 0.70, s)
            rec["gates"] = gates(rec)
            g, f, r = rec["gates"], rec["fab"], rec["raster"]
            angs = sorted(round(float(np.degrees(p.angle))) for p in b["layout"].panels)
            print(f"  {tag:<14} panels {r['n_panels']:>2d} both {r['n_both']:>2d} "
                  f"cross {f['n_usable_crossings']:>2d} slot {f['max_slot_mm']:>5.1f}mm "
                  f"shards {f['n_shards']:>3d} fabIoU {f['iou']:.4f} "
                  f"fabSSIM {f['ssim']:.4f} "
                  f"{'PASS' if all(g.values()) else 'fail:' + ','.join(k for k, v in g.items() if not v)}"
                  f"  angles {angs}", flush=True)
            if all(g.values()):
                save_walls(OUT / tag, "fab", b["targets"], frgb)
            recs[tag] = rec
    ok = {k: v for k, v in recs.items() if all(v["gates"].values())}
    print(f"\n  {len(ok)} of {len(recs)} configurations satisfy all four gates")
    if ok:
        best = max(ok, key=lambda k: ok[k]["fab"]["iou"])
        print(f"  best feasible: {best}  fabIoU {ok[best]['fab']['iou']:.4f}")
        show(best, ok[best])
    _summarise(recs, "bandsweep.json")


# --------------------------------------------------------------------------------------
# PHASE 2 -- systematic parameter sweep
# --------------------------------------------------------------------------------------
# The objective is the WORST wall's fabricated IoU, not the mean.
#
# Every layout in the band sweep was lopsided -- wall B ran 0.93-0.98 while wall A ran
# 0.89. A mean rewards a design that abandons one wall to polish the other, which for an
# installation is the one failure a visitor is guaranteed to notice: they walk around it and
# see the bad side. Optimising the minimum forces the two images to be solved together.
#
# IoU (not SSIM) is primary because the subject is a two-tone MARK: legibility is "is this
# pixel ink or ground", which is exactly what IoU measures. SSIM, MSE and edge fidelity are
# reported alongside as the brief requires, and SSIM is watched for disagreement -- it
# prefers sparser layouts (32-37 shards) where IoU prefers denser ones (50-60).
DEFAULTS = dict(seed=0, panel_count=12, bands="sep60_90", density=0.70,
                ink_t=0.58, tone_bias=0.0, light_r=None, source_radius=None,
                fragment_size=0.135)


def objective(rec) -> float:
    """Worst-wall fabricated MARK IoU; -inf for anything that cannot be built.

    Uses `fab.mark_*` (IoU of the inked region), NOT `fab.iou_*`. The latter comes from
    `logo_render.mark_iou`, which on a grey-on-white mark scores the BACKGROUND and therefore
    barely moves with mark quality. Phases 1-3 were run against that metric before it was
    caught; `recheck()` re-measures the conclusions against this one.

    The NaN guard is not defensive padding: a NaN compares False against everything, so it
    would silently win or lose every `max()` in the search rather than raising. Degenerate
    configurations do occur here (empty subject masks trip `Mean of empty slice` upstream),
    so a non-finite score has to be mapped to "infeasible" explicitly.
    """
    if not all(rec["gates"].values()):
        return float("-inf")
    v = min(rec["fab"]["mark_A"], rec["fab"]["mark_B"])
    return v if np.isfinite(v) else float("-inf")


def _lights(r):
    """Move both lamps along their own axis, keeping the pair symmetric about the diagonal.

    r is the lamp's DISTANCE from the body centre on its principal axis. Nearer lamps mean
    higher magnification (bigger, softer, more overlapped shadows); farther lamps mean a more
    collimated, sharper, smaller projection. Symmetry is kept so neither wall is favoured.
    """
    if r is None:
        return None
    cx = cy = 2.00
    return {"A": (cx + r, cy + 0.55, 1.02), "B": (cx + 0.55, cy + r, 1.02)}


def run_config(ka, kb, **kw) -> tuple:
    """Build + evaluate one point in the design space. Returns (rec, bundle, frgb).

    The target tone TRACKS the ink transmittance (`tone = ink_t + tone_bias`). The first OFAT
    pass held tone at 0.53 while sweeping ink_t, which confounded the axis: it asked for a
    0.53 grey and painted a 0.20 grey, so IoU rose (darker mark separates better from the
    white ground) while SSIM and MSE collapsed. Painting what you asked for is the only
    physically meaningful comparison; `tone_bias` sweeps the deliberate mismatch separately.
    """
    p = dict(DEFAULTS, **kw)
    tone = float(np.clip(p["ink_t"] + p["tone_bias"], 0.05, 0.95))
    solve_kw = dict(fragment_size=float(p["fragment_size"]))
    scene_kw = {}
    if p["source_radius"] is not None:
        scene_kw["source_radius"] = float(p["source_radius"])
    b = build(ka, kb, seed=int(p["seed"]), panel_count=int(p["panel_count"]),
              bands=BAND_SETS[p["bands"]] if isinstance(p["bands"], str) else p["bands"],
              density=float(p["density"]), ink_t=float(p["ink_t"]), tone=tone,
              lights=_lights(p["light_r"]), solve_kw=solve_kw, clip=True)
    rec, _base, frgb, _pieces = evaluate(b, "logo", ka, kb, float(p["density"]), int(p["seed"]))
    rec["gates"] = gates(rec)
    rec["params"] = p
    rec["objective"] = objective(rec)
    return rec, b, frgb


def _row(tag, rec) -> str:
    f, r = rec["fab"], rec["raster"]
    g = rec["gates"]
    return (f"  {tag:<26} mark {rec['objective']:>7.4f}"
            f" (A {f['mark_A']:.4f} B {f['mark_B']:.4f})  bgIoU {f['iou']:.4f}"
            f"  SSIM {f['ssim']:.4f}"
            f"  MSE {f['mse']:.4f}  shards {f['n_shards']:>3d}  min {f['min_diam_mm']:>4.1f}mm"
            f"  both {r['n_both']:>2d}/{r['n_panels']:<2d} cross {f['n_usable_crossings']:>2d}"
            f"  ink {f['ink_area_cm2']:>5.0f}cm2  engr {f['engrave_m']:>5.1f}m"
            f"  {rec['runtime_s']:>4.1f}s"
            f"  {'PASS' if all(g.values()) else 'FAIL:' + ','.join(k for k, v in g.items() if not v)}")


def recheck(seeds=(0, 1, 2)) -> None:
    """Re-measure the Phase 3 conclusion against the CORRECTED metric.

    Phases 1-3 maximised `logo_render.mark_iou`, which scores the white background rather
    than the mark. That does not automatically invalidate the ranking -- background and mark
    agreement are correlated -- but it cannot be assumed either way, so the two endpoints that
    the whole report rests on (the greedy start and the annealing winner) are re-run here and
    compared on both metrics. If the winner no longer leads, the search conclusion is void.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    ka, kb = PAIR
    rows = {}
    for tag, p in (("greedy_start", START), ("anneal_winner", WINNER)):
        for s in seeds:
            rec, _b, _f = run_config(ka, kb, seed=s, **p)
            rows[f"{tag}_s{s}"] = rec
            print(_row(f"{tag}_s{s}", rec), flush=True)
    print("\n  mean over seeds (worst wall):")
    for tag in ("greedy_start", "anneal_winner"):
        rs = [v for k, v in rows.items() if k.startswith(tag)]
        mk = float(np.mean([min(r["fab"]["mark_A"], r["fab"]["mark_B"]) for r in rs]))
        bg = float(np.mean([min(r["fab"]["iou_A"], r["fab"]["iou_B"]) for r in rs]))
        ss = float(np.mean([r["fab"]["ssim"] for r in rs]))
        print(f"    {tag:<16} MARK {mk:.4f}   background {bg:.4f}   SSIM {ss:.4f}")
    json.dump({k: {"fab": v["fab"], "raster": v["raster"], "gates": v["gates"]}
               for k, v in rows.items()},
              open(OUT / "recheck.json", "w"), indent=1, default=str)


def ofat(seeds=(0, 1, 2)) -> None:
    """One-factor-at-a-time screening. Replicated over seeds, because the layout seed was
    already measured to move fab IoU by ~0.03 -- comparable to the factor effects themselves.
    An unreplicated OFAT here would mostly measure seed noise and rank the axes wrongly.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    ka, kb = PAIR
    axes = {
        "ink_t": [0.20, 0.30, 0.40, 0.50, 0.58, 0.65, 0.75],
        "tone_bias": [-0.15, -0.07, 0.0, 0.07, 0.15],
        "bands": ["sep20_40", "sep30_60", "sep60_90", "wide"],
        "panel_count": [6, 8, 10, 12, 14, 16],
        "density": [0.40, 0.55, 0.70, 0.85, 1.00],
        "fragment_size": [0.09, 0.115, 0.135, 0.16, 0.20],
        "source_radius": [0.0005, 0.0015, 0.004, 0.010],
        "light_r": [0.70, 1.00, 1.30, 1.60, 1.90, 2.20],
    }
    # DROPPED, both measured inert over 3 seeds x 4-5 levels (byte-identical objective):
    #   shard_budget -- `face_paper_floor` passes shard_budget=FR.SHARD_BUDGET_PER_WALL to
    #     `fragment_shards_overlap` explicitly, so `scene.overlap_shard_budget` is overridden
    #     and never read. Sweeping it swept nothing.
    #   detail_bias  -- reaches the solver via scene.overlap_detail_bias but moved no metric.
    # Neither is a loss: `face_budget_scan.py` already concluded shard COUNT is set by
    # `density`, not by the budget, which is only a fabrication CEILING. `density` and
    # `fragment_size` are the live controls and are both swept above.
    recs = {}
    for axis, values in axes.items():
        print(f"\n=== {axis} ===", flush=True)
        for v in values:
            objs = []
            for s in seeds:
                tag = f"{axis}={v}_s{s}"
                try:
                    rec, _b, _rgb = run_config(ka, kb, seed=s, **{axis: v})
                except Exception as e:                       # noqa: BLE001 - report, keep sweeping
                    print(f"  {axis}={v} s{s}  CRASH {type(e).__name__}: {e}", flush=True)
                    continue
                recs[tag] = rec
                objs.append(rec["objective"])
                print(_row(tag, rec), flush=True)
            fin = [o for o in objs if np.isfinite(o)]
            print(f"    -> {axis}={v}: mean obj {np.mean(fin):.4f} over {len(fin)}/{len(seeds)}"
                  f" buildable" if fin else f"    -> {axis}={v}: NO buildable seed", flush=True)
    _summarise(recs, "ofat.json")
    _rank(recs, axes)


def _rank(recs, axes) -> None:
    """Effect size per axis: the spread of mean objective across that axis' levels.

    This is what decides where the remaining compute goes. An axis whose best and worst
    levels differ by less than the seed-to-seed spread is not worth optimising.
    """
    print("\n  axis effect sizes (spread of mean objective across levels)")
    eff = {}
    for axis, values in axes.items():
        means = {}
        for v in values:
            o = [r["objective"] for t, r in recs.items()
                 if r["params"][axis] == v and np.isfinite(r["objective"])]
            if o:
                means[v] = float(np.mean(o))
        if len(means) >= 2:
            best = max(means, key=means.get)
            eff[axis] = (max(means.values()) - min(means.values()), best, means[best])
    for axis, (spread, best, val) in sorted(eff.items(), key=lambda kv: -kv[1][0]):
        print(f"    {axis:<14} spread {spread:.4f}   best {axis}={best} at {val:.4f}")


# --------------------------------------------------------------------------------------
# PHASE 3 -- structured optimisation
# --------------------------------------------------------------------------------------
# OFAT measured a real conflict that a single scalar would have buried: pulling the lamps out
# to light_r 1.6 lifts worst-wall fab IoU 0.905 -> 0.945, but drops the panels serving BOTH
# walls from 10/12 to 3/12 and raises ink area 831 -> 2090 cm2.
#
# The mechanism is magnification. Farther lamps mean lower magnification, so a 2 cm shard
# projects to a SMALLER wall feature and the 2 cm floor stops costing anything (raster->fab
# loss falls from -0.045 to -0.018). The same collimation shrinks each panel's shadow, so
# panels stop overlapping and each ends up serving one wall only.
#
# A design with 3/12 shared planes is close to two sculptures sharing a footprint. It still
# clears the >=2 assert, so it is not disqualified -- but it is a different object, and the
# choice belongs to a human. Hence `min_both` is a CONSTRAINT the search runs under, swept
# across its range, rather than a term folded into the objective with an invented weight.
GRID = {
    "ink_t": [0.20, 0.26, 0.32, 0.38, 0.44, 0.50, 0.58, 0.65, 0.72],
    "tone_bias": [-0.12, -0.06, 0.0, 0.06, 0.12],
    "bands": ["sep20_40", "sep30_60", "sep60_90", "wide"],
    "panel_count": [6, 8, 10, 12, 14, 16, 18],
    "density": [0.40, 0.55, 0.70, 0.85, 1.00, 1.15],
    "fragment_size": [0.09, 0.115, 0.135, 0.16, 0.20],
    "source_radius": [0.0005, 0.0015, 0.004, 0.010],
    "light_r": [0.70, 1.00, 1.15, 1.30, 1.45, 1.60, 1.75, 1.90, 2.05, 2.20],
}

# The shipped configuration, used as the common start so every method is compared against the
# design that already exists rather than against a random point.
START = dict(ink_t=0.58, tone_bias=0.0, bands="sep60_90", panel_count=12,
             density=0.70, fragment_size=0.135, source_radius=0.0015, light_r=1.00)

# One start per basin OFAT actually found. `light_r` splits the space into two regimes that a
# single-axis climber cannot travel between (the intermediate radii score worse than both
# ends), so searching from `near` alone silently caps the loose-duty end of the frontier.
STARTS = {
    "near": START,
    "far": dict(START, light_r=1.60, density=0.85, panel_count=14, source_radius=0.010),
}


_REC_CACHE = {}     # (param-key, seed) -> rec.  Shared across every Evaluator in the process.


class Evaluator:
    """Seed-averaged objective with memoisation, so hill climbing and annealing can be
    compared on EVALUATIONS SPENT rather than on wall-clock luck. A revisited point is free,
    which is exactly the accounting a fair comparison of search strategies needs.

    Averaging over seeds is not optional: the layout seed moves the objective by ~0.03,
    comparable to the effect of the factors themselves, so a single-seed search would spend
    its whole budget chasing seed noise.

    The cache holds RECORDS, not scores, and the `min_both` constraint is applied on lookup.
    That lets the frontier sweep re-use one set of solves across every constraint level
    instead of re-solving the same geometry once per level.
    """

    def __init__(self, ka, kb, seeds=(0, 1), min_both=8):
        self.ka, self.kb, self.seeds, self.min_both = ka, kb, tuple(seeds), min_both
        self.seen = set()
        self.n_eval = 0

    @staticmethod
    def key(p):
        return tuple(sorted((k, v) for k, v in p.items() if k != "seed"))

    def _rec(self, k, p, s):
        if (k, s) not in _REC_CACHE:
            try:
                rec, _b, _rgb = run_config(self.ka, self.kb, seed=s, **p)
            except Exception as e:                        # noqa: BLE001 - a dead point, not a stop
                rec = {"_error": f"{type(e).__name__}: {e}"}
            _REC_CACHE[(k, s)] = rec
        return _REC_CACHE[(k, s)]

    def __call__(self, p):
        k = self.key(p)
        if k not in self.seen:
            self.seen.add(k)
            self.n_eval += 1
        vals = []
        for s in self.seeds:
            rec = self._rec(k, p, s)
            if "_error" in rec:
                return float("-inf")
            ok = all(rec["gates"].values()) and rec["raster"]["n_both"] >= self.min_both
            vals.append(rec["objective"] if ok else float("-inf"))
        return float(np.mean(vals)) if all(np.isfinite(vals)) else float("-inf")


def _neighbours(p, rng=None):
    """Adjacent-level moves on the grid: one axis, one step. A dense neighbourhood would make
    hill climbing indistinguishable from random search on a budget this small."""
    out = []
    for axis, levels in GRID.items():
        i = levels.index(p[axis]) if p[axis] in levels else None
        if i is None:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(levels):
                q = dict(p)
                q[axis] = levels[j]
                out.append(q)
    if rng is not None:
        rng.shuffle(out)
    return out


def hill_climb(ev, start, max_iter=12):
    """Steepest-ascent coordinate descent. The BASELINE the brief asks to beat: it is exactly
    what the existing pipeline already does (a greedy choice per axis), just automated."""
    cur, cv = dict(start), ev(dict(start))
    for it in range(max_iter):
        cands = _neighbours(cur)
        scored = [(ev(q), q) for q in cands]
        bv, bq = max(scored, key=lambda t: t[0])
        if bv <= cv + 1e-6:
            print(f"    hill: converged at iter {it}, obj {cv:.4f}, {ev.n_eval} evals", flush=True)
            break
        moved = [k for k in GRID if bq[k] != cur[k]]
        cur, cv = bq, bv
        print(f"    hill iter {it}: obj {cv:.4f}  moved {moved}  {_pstr(cur)}", flush=True)
    return cur, cv


def anneal(ev, start, iters=80, t0=0.020, t1=0.001, seed=0):
    """Simulated annealing over the same grid and the same neighbourhood.

    Hill climbing on this landscape is expected to stall: `light_r` and `density` interact
    (lower magnification wants more ink to cover the same wall), so the single-axis steps that
    reach the good region are individually downhill. Annealing exists here to test that
    claim, not to decorate the report -- if it does not beat the climber, that is the result.
    """
    rng = np.random.default_rng(seed)
    cur, cv = dict(start), ev(dict(start))
    best, bv = dict(cur), cv
    for it in range(iters):
        T = t0 * (t1 / t0) ** (it / max(1, iters - 1))
        cand = _neighbours(cur, rng)[0]
        v = ev(cand)
        if v > cv or (np.isfinite(v) and rng.random() < np.exp((v - cv) / T)):
            cur, cv = cand, v
            if v > bv:
                best, bv = dict(cand), v
                print(f"    anneal iter {it} T {T:.4f}: NEW BEST {bv:.4f}  {_pstr(best)}",
                      flush=True)
    print(f"    anneal: best {bv:.4f}, {ev.n_eval} evals", flush=True)
    return best, bv


def beam(ev, start, width=3, depth=4):
    """Beam search over the same grid and neighbourhood as the other two methods.

    Hill climbing commits to one point and so cannot cross a ridge; annealing can, but only
    stochastically and with no guarantee it revisits the ridge it fell off. Beam search keeps
    `width` incumbents alive simultaneously, which is the cheapest way to carry several
    competing `light_r`/`density` compromises forward at once instead of arbitrating between
    them one step at a time.

    Width and depth are deliberately small. Each evaluation is a full solve over every seed,
    so a wide beam would buy diversity with a budget that the other two methods do not get,
    and the comparison would stop being fair.
    """
    beam_p = [dict(start)]
    best, bv = dict(start), ev(dict(start))
    for d in range(depth):
        seen, cands = set(), []
        for p in beam_p:
            for q in _neighbours(p):
                k = ev.key(q)
                if k not in seen:
                    seen.add(k)
                    cands.append(q)
        if not cands:
            break
        scored = sorted(((ev(q), q) for q in cands), key=lambda t: t[0], reverse=True)
        scored = [t for t in scored if np.isfinite(t[0])][:width]
        if not scored:
            print(f"    beam: no feasible successor at depth {d}", flush=True)
            break
        beam_p = [q for _v, q in scored]
        if scored[0][0] > bv + 1e-6:
            bv, best = scored[0][0], dict(scored[0][1])
            print(f"    beam depth {d}: obj {bv:.4f}  {_pstr(best)}", flush=True)
        else:
            print(f"    beam depth {d}: no improvement (best {bv:.4f})", flush=True)
            break
    print(f"    beam: best {bv:.4f}, {ev.n_eval} evals", flush=True)
    return best, bv


def _pstr(p) -> str:
    return " ".join(f"{k}={p[k]}" for k in
                    ("light_r", "ink_t", "tone_bias", "density", "panel_count", "bands",
                     "fragment_size", "source_radius") if k in p)


def refine(min_both=8, seeds=(0, 1)) -> None:
    """Run both searches from the same start under the same duty constraint, then report the
    duty/fidelity frontier so the constraint itself can be chosen with numbers in hand."""
    OUT.mkdir(parents=True, exist_ok=True)
    ka, kb = PAIR
    start = dict(START)

    print(f"\n=== hill climbing (min_both >= {min_both}) ===", flush=True)
    t0 = time.time()
    ev_h = Evaluator(ka, kb, seeds, min_both)
    p_h, v_h = hill_climb(ev_h, start)
    t_h = time.time() - t0

    print(f"\n=== simulated annealing (min_both >= {min_both}) ===", flush=True)
    t0 = time.time()
    ev_a = Evaluator(ka, kb, seeds, min_both)
    p_a, v_a = anneal(ev_a, start)
    t_a = time.time() - t0

    print(f"\n=== beam search (min_both >= {min_both}) ===", flush=True)
    t0 = time.time()
    ev_b = Evaluator(ka, kb, seeds, min_both)
    p_b, v_b = beam(ev_b, start)
    t_b = time.time() - t0

    print("\n  method comparison (same start, same grid, same constraint)")
    print(f"    {'greedy start':<22} obj {ev_h(start):.4f}")
    print(f"    {'hill climbing':<22} obj {v_h:.4f}  {ev_h.n_eval:>3d} evals  {t_h / 60:.1f} min"
          f"  {_pstr(p_h)}")
    print(f"    {'simulated annealing':<22} obj {v_a:.4f}  {ev_a.n_eval:>3d} evals  {t_a / 60:.1f} min"
          f"  {_pstr(p_a)}")
    print(f"    {'beam search':<22} obj {v_b:.4f}  {ev_b.n_eval:>3d} evals  {t_b / 60:.1f} min"
          f"  {_pstr(p_b)}")

    winner, _wv = max(((p_h, v_h), (p_a, v_a), (p_b, v_b)), key=lambda t: t[1])
    recs = {}
    for s in (0, 1, 2, 3, 4):
        rec, b, frgb = run_config(ka, kb, seed=s, **winner)
        recs[f"best_s{s}"] = rec
        print(_row(f"best_s{s}", rec), flush=True)
        if all(rec["gates"].values()):
            save_walls(OUT / "best", f"s{s}", b["targets"], frgb)
    _summarise(recs, f"refine_both{min_both}.json")
    json.dump({"winner": winner, "hill": [p_h, v_h, ev_h.n_eval],
               "anneal": [p_a, v_a, ev_a.n_eval],
               "beam": [p_b, v_b, ev_b.n_eval]},
              open(OUT / f"refine_both{min_both}_params.json", "w"), indent=1, default=str)


def frontier(levels=(2, 4, 6, 8, 10), seeds=(0, 1)) -> None:
    """Duty/fidelity frontier: the best achievable worst-wall IoU as a function of how many
    planes are required to serve both walls. This is the trade the client actually has to
    decide, so it is reported as a curve, not collapsed into one recommended number.

    MULTI-START, and not as a refinement. The first version climbed from the near-lamp start
    only and returned the SAME point for min_both 2/4/6/8, which reads as "the duty constraint
    is free". It is not: OFAT measured 0.9452 at light_r=1.6 with n_both 3-5, which is feasible
    at min_both <= 4 and beats that point by 0.029. A single-axis climber cannot reach it,
    because the first step from light_r=1.0 toward 1.6 is downhill. Reporting that flat curve
    would have understated the loose-constraint end and hidden the trade entirely.

    Each start is the centre of a basin OFAT actually identified, so this measures the
    frontier rather than the reachability of one arbitrary starting point.
    """
    ka, kb = PAIR
    print("\n  min_both   best obj   params")
    rows = {}
    for mb in levels:
        ev = Evaluator(ka, kb, seeds, mb)
        best_p, best_v, best_from = None, float("-inf"), None
        for name, s0 in STARTS.items():
            p, v = hill_climb(ev, s0, max_iter=8)
            print(f"      [{name}] {v:.4f}", flush=True)
            if v > best_v:
                best_p, best_v, best_from = p, v, name
        rows[mb] = dict(obj=best_v, params=best_p, evals=ev.n_eval, start=best_from)
        print(f"    >= {mb:<5d} {best_v:.4f}   from '{best_from}'   {_pstr(best_p)}", flush=True)
    json.dump(rows, open(OUT / "frontier.json", "w"), indent=1, default=str)


def _summarise(recs, fname) -> None:
    (OUT / fname).write_text(json.dumps(
        {k: {"raster": v["raster"], "fab": v["fab"], "gates": v["gates"],
             "d_ssim": v["d_ssim"], "d_iou": v["d_iou"], "runtime_s": v["runtime_s"]}
         for k, v in recs.items()}, indent=2), encoding="utf-8")

    print("\n" + "=" * 112)
    print(f"{'config':<20} {'rasSSIM':>8} {'fabSSIM':>8} {'dSSIM':>8} {'rasIoU':>7} {'fabIoU':>7} "
          f"{'dIoU':>7} {'shards':>6} {'minmm':>6} {'both':>6} {'cross':>5} {'gates':>6}")
    print("-" * 112)
    for tag, rec in recs.items():
        r, f, g = rec["raster"], rec["fab"], rec["gates"]
        print(f"{tag:<20} {r['ssim']:>8.4f} {f['ssim']:>8.4f} {rec['d_ssim']:>+8.4f} "
              f"{r['iou']:>7.4f} {f['iou']:>7.4f} {rec['d_iou']:>+7.4f} "
              f"{f['n_shards']:>6d} {f['min_diam_mm']:>6.1f} "
              f"{r['n_both']:>3d}/{r['n_panels']:<2d} {f['n_usable_crossings']:>5d} "
              f"{sum(g.values()):>4d}/4")
    print("=" * 112)
    print(f"wrote {OUT / fname}")


# --------------------------------------------------------------------------------------
# PHASE 5 -- verify, export, ship
# --------------------------------------------------------------------------------------
# The result of the Phase 3 search (simulated annealing, min_both >= 8, seeds 0-1).
# Kept as a literal so the shipped piece is reproducible without re-running a 30 min search.
WINNER = dict(ink_t=0.58, tone_bias=0.0, bands="sep60_90", panel_count=16,
              density=1.00, fragment_size=0.135, source_radius=0.010, light_r=1.00)
SHIP_SEED = 3          # best worst-wall fab IoU (0.9239) of seeds 0-4, all of which passed


def verify(rec, pieces) -> None:
    """Hard asserts. These raise rather than warn because every one of them is a claim made to
    a fabricator or to the viewer, and a silently-violated claim becomes a scrapped sheet of
    Perspex or a sculpture that does not show the second image.

    The 2 cm test is re-run here on the EXPORTED polygons, not read back from the report, so
    it certifies the objects actually written to cut/ rather than an earlier copy of them.
    """
    f, r = rec["fab"], rec["raster"]
    polys = [p for items in pieces.values() for (p, _c, _s) in items]

    worst = min((inscribed_diameter(p) for p in polys), default=0.0)
    assert polys, "no shards at all"
    assert worst >= MIN_SHARD - 1e-6, (
        f"a shard of {worst * 1000:.2f} mm reached the cut files; the floor is "
        f"{MIN_SHARD * 1000:.0f} mm and the laser will not be re-run for free")

    assert f["n_usable_crossings"] >= 1, (
        "no usable weave crossing: the planes do not physically intersect, so this is a rack "
        "of parallel sheets and the 'two intersecting planes' requirement is unmet")

    # The projection must be REAL: at least two planes must be ABLATION-VERIFIED to change
    # both walls. `n_both` is computed by knocking each plane out and measuring the damage to
    # each image, so a plane counts only if removing it degrades A *and* B.
    assert r["n_both"] >= 2, (
        f"only {r['n_both']} plane(s) serve both walls; without at least two this is two "
        f"separate sculptures sharing a footprint, not one dual-image piece")

    assert f["fits_body"], (
        f"envelope {f['env_x_m']:.2f} x {f['env_y_m']:.2f} x {f['env_z_m']:.2f} m exceeds the "
        f"{BODY:.2f} m body")

    assert max(c.slot_width for c in usable_crossings(rec["_layout"])) <= 0.010, \
        "a weave slot wider than 10 mm is a gap, not a joint"

    print(f"  VERIFIED  min shard {worst * 1000:.1f} mm >= {MIN_SHARD * 1000:.0f} mm"
          f" | {r['n_both']} planes serve BOTH walls (ablation-verified, need >= 2)"
          f" | {f['n_usable_crossings']} real crossings"
          f" | envelope {f['env_x_m']:.2f} x {f['env_y_m']:.2f} x {f['env_z_m']:.2f} m")


def ship(seed=SHIP_SEED, params=None, out=None) -> dict:
    """Build the chosen design once, verify it, and write everything a fabricator needs.

    The cut files are exported from the SHARD-FLOOR-FILTERED pieces -- the same object that
    was scored -- so the DXF/SVG describe the piece the reported numbers refer to. Exporting
    the unfiltered `panel_stack_pieces` instead would quietly reintroduce the sub-2 cm shards
    that the whole Phase 0 harness exists to catch.
    """
    from shadowart.preview.interactive3d import build_interactive

    p = dict(WINNER if params is None else params)
    out = OUT / "ship" if out is None else Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ka, kb = PAIR
    print(f"\n{'=' * 100}\n=== SHIP  {ka} x {kb}  seed={seed}  {_pstr(p)}\n{'=' * 100}")

    rec, b, frgb = run_config(ka, kb, seed=seed, **p)
    pieces, _fcid, _pT, _rgb, _dropped = fab_render(b)
    rec["_layout"] = b["layout"]
    print(_row("shipped", rec))
    show("shipped", rec)
    verify(rec, pieces)
    del rec["_layout"]

    # Walls: the FABRICATED render, not the raster field -- this is what will be on the wall.
    save_walls(out, "fab", b["targets"], frgb)
    for w in ("A", "B"):
        Image.fromarray((np.clip(np.flipud(b["targets"][w]), 0, 1) * 255).astype(np.uint8)) \
            .convert("RGB").save(out / f"target_{w}.jpg", quality=95)
        Image.fromarray((np.clip(np.flipud(frgb[w]), 0, 1) * 255).astype(np.uint8)) \
            .convert("RGB").save(out / f"wall_{w}.jpg", quality=95)

    poly_channel = {id(pl): ch for items in pieces.values() for pl, ch, _s in items}
    flat = {n: [pl for pl, _c, _s in items] for n, items in pieces.items()}
    build_interactive(b["layout"], b["table"], b["opacity"], None,
                      out / "scene_interactive.html", rays=40, auto_open=False,
                      wall_rgb=frgb, pieces=flat,
                      color_of=lambda panel, poly: tuple(
                          C.display_rgb(poly_channel.get(id(poly), "clear"))))

    fab = LF.export_fab(out, b["layout"], pieces, b["stack_colorid"], b["names"])

    panels = [dict(name=pn.name,
                   angle_deg=float(np.degrees(pn.angle)) % 180.0,
                   u_range=[float(x) for x in pn.u_range],
                   v_range=[float(x) for x in pn.v_range],
                   u_size_m=float(pn.u_size), v_size_m=float(pn.v_size),
                   anchor=[float(x) for x in pn.anchor],
                   floor_xy=[[float(c) for c in pt] for pt in pn.floor_segment_xy()])
              for pn in b["layout"].panels]
    crossings = [dict(a=c.panelA_name, b=c.panelB_name, x=float(c.x), y=float(c.y),
                      z_range=[float(z) for z in c.z_range], angle_deg=float(c.angle_deg),
                      slot_mm=float(c.slot_width * 1000))
                 for c in usable_crossings(b["layout"])]
    rec["export"] = fab
    rec["panels"] = panels
    rec["crossings"] = crossings
    (out / "metrics.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {out}/  (cut/, shards.obj, shards.ply, scene_interactive.html,"
          f" wall_*.jpg, target_*.jpg, metrics.json)")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="Phase 0: validate the fab harness")
    ap.add_argument("--baseline", action="store_true", help="Phase 1: cost of the constraints")
    ap.add_argument("--feasible", action="store_true", help="Phase 1b: seeds that pass all gates")
    ap.add_argument("--bands", action="store_true", help="Phase 2a: panel orientation sweep")
    ap.add_argument("--ofat", action="store_true", help="Phase 2b: one-factor-at-a-time screening")
    ap.add_argument("--refine", action="store_true", help="Phase 3: hill climbing vs annealing")
    ap.add_argument("--frontier", action="store_true", help="Phase 3b: duty/fidelity frontier")
    ap.add_argument("--ship", action="store_true", help="Phase 5: verify + export the winner")
    ap.add_argument("--recheck", action="store_true", help="re-measure vs the corrected metric")
    ap.add_argument("--seed", type=int, default=SHIP_SEED, help="layout seed for --ship")
    ap.add_argument("--min-both", type=int, default=8, help="planes required to serve BOTH walls")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.baseline:
        baseline()
    elif a.feasible:
        feasible()
    elif a.bands:
        bandsweep()
    elif a.ofat:
        ofat()
    elif a.refine:
        refine(min_both=a.min_both)
    elif a.frontier:
        frontier()
    elif a.ship:
        ship(seed=a.seed)
    elif a.recheck:
        recheck()
    else:
        selftest()
        baseline()
        bandsweep()
        ofat()
        refine(min_both=a.min_both)


if __name__ == "__main__":
    main()
