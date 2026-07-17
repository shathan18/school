"""Shard-edge thickness shadow -- actually simulated, not just estimated/reported.

Real material has thickness, so a shard's CUT EDGE is a narrow strip of acrylic seen
close to edge-on. At a shallow enough viewing angle from the light, that edge blocks
(or at least dims) light beyond the shard's own flat footprint, extending its cast
shadow by roughly

    w = thickness / tan(grazing_angle)

where `grazing_angle` is measured from the panel's own plane (0 = the light rays skim
exactly along the surface -> the edge reads full-on, unboundedly wide; pi/2 = normal
incidence -> the edge is invisible, width 0). This module bakes that width directly
into each shard's rasterised footprint (grown outward by nearest-label dilation) BEFORE
the existing homography-warp + penumbra-blur render path, so it genuinely changes
`pred_rgb` and therefore RMSE/SSIM/PSNR/edge-fidelity computed against it -- not a
caveat reported alongside an otherwise-unmodified render.

Scope: mode (a) only (the natural/minimized width from the design's actual geometry).
Mode (b) (a forced stained-glass/lead-came minimum width) was cut from this pass per
explicit instruction to trim time from the thickness-simulation work rather than the
acrylic-thickness comparison; the mechanism below supports it trivially (swap the width
formula for a fixed floor) if it's wanted later.

Implementation note: "grow a shard's mask outward by w, using an outline stroke added
ON TOP of the shard's own already-opaque footprint" is exactly what a morphological
dilate(mask, w) produces (a dilate-minus-erode boundary band unioned back onto the
original mask collapses to the same set, since erode(mask) subset mask subset
dilate(mask) for any structuring element containing the origin) -- so the widened
footprint is computed directly via dilation instead of building and re-unioning a
separate boundary band.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from ..geometry.projection import primary_wall_of


def grazing_angle_rad(panel, light_xyz):
    """Angle between the incoming ray (light -> panel centre) and the panel's own
    PLANE (not its normal): 0 = ray skims along the surface, pi/2 = hits it face-on."""
    c_uv = panel.corners_uv().mean(axis=0)
    P = panel.uv_to_xyz(c_uv[None, :])[0]
    d = np.asarray(light_xyz, float) - P
    dn = d / max(np.linalg.norm(d), 1e-9)
    cos_incidence = float(np.clip(abs(dn @ panel.normal), 0.0, 1.0))
    return math.asin(cos_incidence)


def _dilate_labels(colorid_2d, w_px, intensity_2d=None):
    """Grow every labelled shard region outward by `w_px` pixels, each new pixel taking
    its NEAREST shard's own label -- so two different shards' edge-shadows don't bleed
    into and relabel each other beyond the dilation radius. `intensity_2d` (optional) is
    grown with the SAME nearest-source mapping so the intensity-weighted tint stays aligned
    with the colour-id in every grown edge pixel (otherwise the new edge band would carry a
    colour-id but zero intensity -> render as an untinted/clear halo)."""
    if w_px <= 0:
        return (colorid_2d, intensity_2d)
    mask = colorid_2d > 0
    if not mask.any():
        return (colorid_2d, intensity_2d)
    grown = ndimage.binary_dilation(mask, iterations=w_px)
    inds = ndimage.distance_transform_edt(~mask, return_indices=True, return_distances=False)
    idx = tuple(inds)
    out_c = np.where(grown, np.where(mask, colorid_2d, colorid_2d[idx]), 0).astype(colorid_2d.dtype)
    if intensity_2d is None:
        return (out_c, None)
    out_i = np.where(grown, np.where(mask, intensity_2d, intensity_2d[idx]), 0.0).astype(intensity_2d.dtype)
    return (out_c, out_i)


def edge_shadow_width_m(panel, scene, table, min_tan=0.05, thickness_m=None):
    """Natural (mode-a) edge-shadow width for `panel`, using ITS OWN primary wall's
    light for the grazing angle -- the renderer warps one physical opacity field per
    panel onto both walls, so one width is baked in per panel rather than two (see
    module docstring); using the primary-wall geometry is the panel's dominant, most
    visible role. `min_tan` floors the divide-by-near-zero case (a panel viewed almost
    exactly edge-on to its own light) at a physically sane cap rather than letting the
    shadow width run to infinity. `thickness_m` overrides `panel.thickness` (for a
    thickness sweep without rebuilding the scene)."""
    wall_name = primary_wall_of(scene, table, panel)
    light_xyz = scene.light_for_wall(wall_name).xyz
    g = grazing_angle_rad(panel, light_xyz)
    tan_g = max(math.tan(g), min_tan)
    t = panel.thickness if thickness_m is None else thickness_m
    return t / tan_g


def apply_thickness_shadow(scene, table, panels, stack_colorid, stack_intensity=None,
                           min_tan=0.05, thickness_m=None):
    """Return NEW (stack_colorid, stack_intensity) with every shard's footprint on every
    panel grown outward by that panel's natural edge-shadow width, in panel pixels. The
    intensity array is grown with the same nearest-source mapping so the tint stays aligned
    (see `_dilate_labels`). Leaves panels whose computed width rounds to <1px untouched.
    `thickness_m` overrides the material thickness for a sweep; if None, uses panel.thickness."""
    out_c = stack_colorid.copy()
    out_i = None if stack_intensity is None else stack_intensity.copy()
    S, P, Hp, Wp = stack_colorid.shape
    for gi, panel in enumerate(panels):
        width_m = edge_shadow_width_m(panel, scene, table, min_tan, thickness_m=thickness_m)
        px_m = 0.5 * (panel.u_size / Wp + panel.v_size / Hp)
        w_px = max(0, int(round(width_m / max(px_m, 1e-9))))
        if w_px <= 0:
            continue
        for s in range(S):
            i_slot = None if out_i is None else stack_intensity[s, gi]
            c_new, i_new = _dilate_labels(stack_colorid[s, gi], w_px, i_slot)
            out_c[s, gi] = c_new
            if out_i is not None:
                out_i[s, gi] = i_new
    return out_c, out_i
