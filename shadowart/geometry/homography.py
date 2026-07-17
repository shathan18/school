"""Point-light central projection between two planes = a projective homography.

A point light L, a source plane (a panel) and a destination plane (a wall) define a
collineation: the shadow of a panel point P is where the ray L->P meets the wall.
For two *parallel* planes (family-A panel & Wall A) this reduces to a uniform scaling
by the magnification m = dist(L, wall)/dist(L, panel); for perpendicular planes
(family-A panel casting onto Wall B) it is a strongly keystoned homography — which is
exactly the cross-talk. We compute the full 3x3 homography for every (panel, light,
wall) triple so both cases are handled by one code path.
"""
from __future__ import annotations

import numpy as np


def ray_plane_intersect(L, P, plane_pt, plane_n):
    """Intersect the ray from L through P with plane {x : n.(x-plane_pt)=0}.

    Returns (X, t) where X = L + t*(P-L), or (None, None) if the ray is parallel.
    """
    L = np.asarray(L, float); P = np.asarray(P, float)
    plane_pt = np.asarray(plane_pt, float); plane_n = np.asarray(plane_n, float)
    d = P - L
    denom = plane_n @ d
    if abs(denom) < 1e-12:
        return None, None
    t = (plane_n @ (plane_pt - L)) / denom
    return L + t * d, t


def project_uv_to_wall(panel, light_xyz, wall, uv):
    """Project panel-local (u,v) metres -> wall-local (a,b) metres through the light.

    uv: (...,2) array.  Returns (...,2) wall coords; points whose ray misses the wall
    plane (parallel) come back as NaN.
    """
    uv = np.asarray(uv, float)
    flat = uv.reshape(-1, 2)
    P = panel.uv_to_xyz(flat)                       # (N,3)
    L = np.asarray(light_xyz, float)
    n = wall.normal; p0 = wall.origin
    out = np.full((flat.shape[0], 2), np.nan)
    d = P - L
    denom = d @ n
    ok = np.abs(denom) > 1e-12
    t = np.where(ok, (n @ (p0 - L)) / np.where(ok, denom, 1.0), np.nan)
    X = L[None, :] + t[:, None] * d                 # (N,3)
    rel = X - wall.origin
    a = rel @ wall.axis_u
    b = rel @ wall.axis_v
    out[:, 0] = np.where(ok, a, np.nan)
    out[:, 1] = np.where(ok, b, np.nan)
    return out.reshape(uv.shape)


def dlt_homography(src, dst):
    """3x3 homography H such that dst ~ H @ [src;1], from >=4 correspondences.

    Uses isotropic normalisation for numerical stability, then DLT via SVD.
    """
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    Ts = _normalise(src); Td = _normalise(dst)
    s = _apply_h(Ts, src); d = _apply_h(Td, dst)
    n = src.shape[0]
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = s[i]; u, v = d[i]
        A[2 * i]     = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts                  # de-normalise
    return H / H[2, 2]


def _normalise(pts):
    """Similarity transform sending the centroid to 0 and mean dist to sqrt(2)."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d if d > 1e-12 else 1.0
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def _apply_h(H, pts):
    """Apply homogeneous H to `pts`, dividing by the projective w-coordinate.

    Guards against an exact (or near-exact) zero w -- a real, if rare, geometric
    degeneracy (a ray exactly parallel to the target plane), not a bug: e.g. a light
    positioned so its own in-plane coordinate exactly coincides with a sampled pixel
    column/row makes every ray from it to that column stay parallel to any panel plane
    that isn't literally at that same coordinate, for every point along that column.
    Clipping the denominator's magnitude away from zero sends the result to a very
    large (but finite) coordinate instead of NaN -- grid_sample's zero-padding then
    correctly reads it as "off panel" (no contribution), which is the physically
    sensible answer: no real, finite point on that panel actually casts a shadow at a
    wall location its rays never reach."""
    pts = np.asarray(pts, float)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    out = hom @ H.T
    w = out[:, 2:3]
    tiny = np.abs(w) < 1e-9
    if np.any(tiny):
        w = np.where(tiny, 1e-9, w)
    return out[:, :2] / w


def apply_homography(H, pts):
    """Apply 3x3 H to (...,2) points; returns (...,2)."""
    pts = np.asarray(pts, float)
    shape = pts.shape
    flat = pts.reshape(-1, 2)
    out = _apply_h(H, flat)
    return out.reshape(shape)


def homography_panel_to_wall(panel, light_xyz, wall):
    """3x3 homography mapping panel-local (u,v) metres -> wall-local (a,b) metres."""
    src = panel.corners_uv()                          # 4 corners in panel metres
    dst = project_uv_to_wall(panel, light_xyz, wall, src)
    if np.any(~np.isfinite(dst)):
        raise ValueError(f"panel {panel.name}: shadow rays do not all meet wall {wall.name} "
                         "(check light position / panel orientation).")
    return dlt_homography(src, dst)


def magnification(panel, light_xyz, wall):
    """Along-ray magnification at the panel centre (shadow size / panel feature size).

    For parallel panel/wall this is exactly dist(L,wall)/dist(L,panel) along the normal.
    """
    c_uv = panel.corners_uv().mean(axis=0)
    P = panel.uv_to_xyz(c_uv[None, :])[0]
    _, t = ray_plane_intersect(light_xyz, P, wall.origin, wall.normal)
    return abs(t) if t is not None else np.inf
