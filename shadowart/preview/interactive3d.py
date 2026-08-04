"""Interactive 3D preview (Plotly -> self-contained HTML you can orbit/zoom/pan).

Shows the shards floating in the corner volume, both walls with the predicted projection,
the floor lights and (optionally) light rays. In colour mode each shard is tinted by its
perspex colour and the walls show the predicted colour image; otherwise shards are tinted by
depth plane and the walls show scalar darkness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import matplotlib
import matplotlib.cm as cm

from ..geometry import homography as H
from ..raster2vec import halftone, features, contours
from ..fabricate.shard_geom import build_shard_mesh
from ..fabricate.export_color import _piece_color
from ..targets import color as _color


def pieces_from_opacity(scene, opacity):
    out = {}
    for pi, panel in enumerate(scene.panels):
        Hp, Wp = opacity[pi].shape
        su = (panel.u_range[1] - panel.u_range[0]) / Wp
        sv = (panel.v_range[1] - panel.v_range[0]) / Hp
        mask = halftone.threshold(opacity[pi], scene.fab.threshold)
        mask = features.enforce_min_feature(mask, scene.fab.min_feature, 0.5 * (su + sv))
        out[panel.name] = contours.mask_to_polygons(mask, panel.u_range, panel.v_range)
    return out


# --- shard colouring -------------------------------------------------------
def _depth_color_of(scene, table):
    """Colour by how 'deep' (far from its light) a shard's host panel is, relative to
    other panels currently serving the SAME wall -- magnification is a natural proxy
    (farther panels magnify more), grouped by `primary_wall_of` since there's no family
    to group panels by directly. Replaces the old family-list-index gradient position."""
    from ..geometry.projection import primary_wall_of
    # matplotlib >= 3.9 removed cm.get_cmap; the colormaps registry is the modern, version-safe path.
    turbo = matplotlib.colormaps["turbo"] if hasattr(matplotlib, "colormaps") else cm.get_cmap("turbo")
    groups = {}
    for panel in scene.panels:
        groups.setdefault(primary_wall_of(scene, table, panel), []).append(panel)
    for members in groups.values():
        members.sort(key=lambda p: table[(p.name, primary_wall_of(scene, table, p))].magnification)
    rank = {p.name: i for members in groups.values() for i, p in enumerate(members)}
    sizes = {w: len(members) for w, members in groups.items()}

    def f(panel, poly):
        w = primary_wall_of(scene, table, panel)
        dr = rank[panel.name] / max(sizes[w] - 1, 1)
        return turbo(0.12 + 0.72 * dr)[:3]
    return f


def _palette_color_of(scene, colorid, names):
    idx = {p.name: i for i, p in enumerate(scene.panels)}

    def f(panel, poly):
        name = _piece_color(panel, poly, colorid[idx[panel.name]], names)
        return tuple(_color.display_rgb(name))
    return f


def _shard_trace(scene, pieces, thickness, color_of):
    verts, faces, colors = build_shard_mesh(scene, pieces, thickness, color_of)
    if len(verts) == 0:
        return None
    # Mesh3d's `vertexcolor` needs a list of colour STRINGS, not a raw [N,3] numeric
    # array -- passing the array directly (the old code) makes plotly serialise it as
    # generic numeric `bdata` with no colourscale, which renders every shard flat
    # white/blank regardless of its actual per-shard colour. `_wall_rgb_mesh` below
    # already gets this right (`facecolor=[f"rgb(...)" ...]`); match that here.
    vc = np.clip(colors * 255, 0, 255).astype(int)
    vertexcolor = [f"rgb({r},{g},{b})" for r, g, b in vc]
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                     vertexcolor=vertexcolor, opacity=0.85, flatshading=True, name="shards",
                     showlegend=True, lighting=dict(ambient=0.55, diffuse=0.7, specular=0.3))


# --- walls -----------------------------------------------------------------
def _wall_surface(scene, wall_name, pred):
    wall = scene.walls[wall_name]
    Hn, Wn = pred[wall_name].shape
    a = np.linspace(0, wall.width, Wn); b = np.linspace(0, wall.height, Hn)
    A, B = np.meshgrid(a, b)
    o, eu, ev = wall.origin, wall.axis_u, wall.axis_v
    return go.Surface(x=o[0] + A * eu[0] + B * ev[0], y=o[1] + A * eu[1] + B * ev[1],
                      z=o[2] + A * eu[2] + B * ev[2], surfacecolor=pred[wall_name],
                      cmin=0, cmax=1, showscale=False,
                      colorscale=[[0, "rgb(248,248,248)"], [1, "rgb(15,15,15)"]],
                      name=f"Wall {wall_name}", showlegend=True,
                      lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0))


def _wall_rgb_mesh(scene, wall_name, rgb, gs=90):
    """Colour projection on a wall as a grid Mesh3d. Uses per-FACE colour (rgb strings) —
    plotly does not apply vertexcolor reliably on a perfectly flat/coplanar mesh."""
    wall = scene.walls[wall_name]
    Hn, Wn = rgb.shape[:2]
    rows = np.linspace(0, Hn - 1, gs).astype(int)
    cols = np.linspace(0, Wn - 1, gs).astype(int)
    a = np.linspace(0, wall.width, gs); b = np.linspace(0, wall.height, gs)
    A, B = np.meshgrid(a, b)
    o, eu, ev = wall.origin, wall.axis_u, wall.axis_v
    X = o[0] + A * eu[0] + B * ev[0]; Y = o[1] + A * eu[1] + B * ev[1]; Z = o[2] + A * eu[2] + B * ev[2]
    col = (np.clip(rgb[np.ix_(rows, cols)].reshape(-1, 3), 0, 1) * 255).astype(int)
    faces = []
    for r in range(gs - 1):
        for c in range(gs - 1):
            i0 = r * gs + c
            faces += [(i0, i0 + 1, i0 + gs + 1), (i0, i0 + gs + 1, i0 + gs)]
    faces = np.asarray(faces)
    fc = col[faces].mean(axis=1).astype(int)               # per-face colour
    return go.Mesh3d(x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                     facecolor=[f"rgb({r},{g},{b})" for r, g, b in fc],
                     name=f"Wall {wall_name}", showlegend=True, flatshading=True,
                     lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0))


def _rays(scene, table, pieces, n):
    """Draw each sampled shard's ray toward whichever wall it currently contributes the
    most to (`primary_wall_of` -- no family to read this from directly)."""
    from ..geometry.projection import primary_wall_of
    Xr, Yr, Zr = [], [], []
    shards = [(p, poly) for p in scene.panels for poly in pieces.get(p.name, [])]
    if not shards:
        return None
    step = max(1, len(shards) // max(n, 1))
    for panel, poly in shards[::step]:
        c3d = panel.uv_to_xyz(np.array(poly.centroid.coords[0])[None, :])[0]
        wall_name = primary_wall_of(scene, table, panel)
        L = scene.light_for_wall(wall_name).xyz
        wall = scene.walls[wall_name]
        hit, t = H.ray_plane_intersect(L, c3d, wall.origin, wall.normal)
        if hit is None or t is None or t < 1:
            continue
        for pt in (L, c3d, hit):
            Xr.append(pt[0]); Yr.append(pt[1]); Zr.append(pt[2])
        Xr.append(None); Yr.append(None); Zr.append(None)
    return go.Scatter3d(x=Xr, y=Yr, z=Zr, mode="lines", name="light rays",
                        line=dict(color="rgba(240,200,60,0.35)", width=2),
                        showlegend=True, hoverinfo="skip")


def _lights_floor(scene):
    L = np.array([lt.xyz for lt in scene.lights.values()])
    traces = [go.Scatter3d(x=L[:, 0], y=L[:, 1], z=L[:, 2], mode="markers+text",
                           text=[f"L{n}" for n in scene.lights], textposition="top center",
                           marker=dict(size=6, color="gold", symbol="diamond"), name="lights")]
    # floor extent covers both lights and every panel's floor-plan footprint --
    # no family/.coord to read, so every panel's own floor_segment_xy() endpoints
    # are checked instead (same approach as fabricate/export_obj.py::_floor_extent).
    xs = [L[:, 0].max(), 0.0]; ys = [L[:, 1].max(), 0.0]
    for p in scene.panels:
        for (x, y) in p.floor_segment_xy():
            xs.append(x); ys.append(y)
    xmax = float(max(xs) * 1.1)
    ymax = float(max(ys) * 1.1)
    traces.append(go.Mesh3d(x=[0, xmax, xmax, 0], y=[0, 0, ymax, ymax], z=[0, 0, 0, 0],
                            i=[0, 0], j=[1, 2], k=[2, 3], color="rgb(220,220,225)",
                            opacity=0.25, name="floor", hoverinfo="skip", showlegend=False))
    return traces


def _box_mesh(x0, x1, y0, y1, z0, z1, color, name, opacity=0.9):
    """Axis-aligned solid box as a single Mesh3d (8 vertices, 12 triangles)."""
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
    j = [1, 2, 5, 6, 1, 5, 3, 7, 3, 7, 2, 6]
    k = [2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5]
    return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color, opacity=opacity,
                     flatshading=True, name=name, hoverinfo="skip", showlegend=False)


def _table_traces(table):
    """The display table under the shard body (scene.table, visualisation only):
    a top slab plus optional legs down to the floor."""
    cx, cy = table.center
    sx, sy = table.size
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    wood = "rgb(150,110,75)"
    traces = [_box_mesh(x0, x1, y0, y1, table.top_z - table.thickness, table.top_z,
                        wood, "table")]
    if table.legs:
        leg, inset = 0.05, 0.06
        for lx, ly in ((x0 + inset, y0 + inset), (x1 - inset - leg, y0 + inset),
                       (x0 + inset, y1 - inset - leg), (x1 - inset - leg, y1 - inset - leg)):
            traces.append(_box_mesh(lx, lx + leg, ly, ly + leg, 0.0,
                                    table.top_z - table.thickness, wood, "table leg"))
    return traces


def build_interactive(scene, table, opacity, pred, out_path, rays=40, auto_open=True,
                      show_panels=False, shard_thickness=0.02,
                      panel_colorid=None, names=None, wall_rgb=None, pieces=None,
                      color_of=None, embed_plotly=False, show_walls=True):
    if pieces is None:
        pieces = pieces_from_opacity(scene, opacity)
    if color_of is None:
        color_mode = panel_colorid is not None
        color_of = (_palette_color_of(scene, panel_colorid, names) if color_mode
                    else _depth_color_of(scene, table))
    traces = []
    if show_walls:
        for wname in scene.walls:                          # one wall per turntable stop
            traces.append(_wall_rgb_mesh(scene, wname, wall_rgb[wname]) if wall_rgb is not None
                          else _wall_surface(scene, wname, pred))
    sm = _shard_trace(scene, pieces, shard_thickness, color_of)
    if sm is not None:
        traces.append(sm)
    if rays:
        rr = _rays(scene, table, pieces, rays)
        if rr is not None:
            traces.append(rr)
    traces += _lights_floor(scene) if show_walls else []
    if getattr(scene, "table", None) is not None:
        traces += _table_traces(scene.table)

    fig = go.Figure(data=traces)
    # `aspectmode="data"` keeps real proportions, and the rig is far wider than it is tall, so
    # the camera eye sits in a badly flattened box: z=1.3 against a z-extent of ~0.3 opens the
    # scene looking almost straight down, with all three wall images edge-on. Drop the eye to
    # roughly standing height instead, which is where the piece is meant to be seen from.
    fig.update_layout(
        title="ShadowArt — interactive 3D (drag to orbit, scroll to zoom)",
        scene=dict(aspectmode="data", xaxis_title="x (m)",
                   yaxis_title="y (m)", zaxis_title="z (up)",
                   camera=dict(eye=dict(x=1.35, y=1.35, z=0.35),
                               center=dict(x=0, y=0, z=0))),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), auto_open=auto_open,
                  include_plotlyjs=(True if embed_plotly else "cdn"))
    return str(out_path), fig
