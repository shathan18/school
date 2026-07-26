"""Fragment the image into a "cryptic cloud" of shards spread across the depth planes.

Instead of one contiguous ~1/n slab per panel (which still reads as the object), we
shatter each wall's silhouette into many small ORGANIC shards and scatter them evenly
across the depth planes currently relevant to that wall (`primary_wall_of` -- no panel
declares which wall it's "for", it's computed from projected geometry):

  - blue-noise seeds inside the silhouette -> nearest-seed (Voronoi) cells = shards,
  - shard size is capped (big cells re-split, tiny slivers dropped),
  - each shard is assigned to one depth plane (balanced, optional near-bias),
  - shards are eroded by a small gap so they read as separated / floating,
  - the union across all planes still tiles the silhouette (IoU stays high),
  - so any single plane is unreadable dust; only the shadow resolves the image.

Finally, genuine physical collisions where two panels cross are removed by trimming a
thin clearance notch from one side.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..geometry import homography as H
from ..geometry.panels import weave_crossings
from ..geometry.projection import primary_wall_of
from ..targets import color as _color
from .initializer import _panel_pixel_uv


# ---------------------------------------------------------------------------
# seeding + fragmentation (wall space)
# ---------------------------------------------------------------------------
def _seed_points(mask, spacing_px, rng, jitter=0.7):
    """Jittered-grid (blue-noise-ish) seeds that fall inside `mask`. Returns (K,2) row,col."""
    Hn, Wn = mask.shape
    s = max(2.0, spacing_px)
    seeds = []
    gy = np.arange(s / 2, Hn, s)
    gx = np.arange(s / 2, Wn, s)
    for cy in gy:
        for cx in gx:
            iy = int(round(cy + (rng.random() - 0.5) * jitter * s))
            ix = int(round(cx + (rng.random() - 0.5) * jitter * s))
            if 0 <= iy < Hn and 0 <= ix < Wn and mask[iy, ix]:
                seeds.append((iy, ix))
    if not seeds:                                    # tiny silhouette -> use its centroid
        ys, xs = np.where(mask)
        if len(ys):
            seeds.append((int(ys.mean()), int(xs.mean())))
    return np.asarray(seeds, dtype=float)


def _voronoi_labels(mask, seeds):
    """Assign every True pixel of `mask` to its nearest seed. Returns int label image (-1 out)."""
    ys, xs = np.where(mask)
    lab = np.full(mask.shape, -1, dtype=int)
    if len(seeds) == 0 or len(ys) == 0:
        return lab
    _, idx = cKDTree(seeds).query(np.stack([ys, xs], axis=1))
    lab[ys, xs] = idx
    return lab


def _local_cap(region, max_px, importance, detail_bias):
    """Effective size cap for `region`: shrinks where `importance` runs high there, so
    detailed areas (edges/colour transitions) get subdivided into more, smaller shards while
    flat areas stay coarse -- same total splitting machinery, just a spatially-varying
    trigger instead of one global size. `importance=None` reproduces the original constant
    cap exactly (backward compatible for callers that don't pass it)."""
    if importance is None or detail_bias <= 0:
        return max_px
    imp = float(importance[region].mean()) if region.any() else 0.0
    return max_px / (1.0 + detail_bias * imp)


def _split_large(comp, max_px, rng, depth=0, importance=None, detail_bias=0.0):
    """Recursively split a too-big (or too-detailed) component into sub-cells."""
    cap = _local_cap(comp, max_px, importance, detail_bias)
    area = int(comp.sum())
    if area <= cap or depth >= 3:
        return [comp]
    sub_s = np.sqrt(cap) * 0.85
    seeds = _seed_points(comp, sub_s, rng)
    if len(seeds) < 2:
        return [comp]
    lab = _voronoi_labels(comp, seeds)
    out = []
    for k in range(len(seeds)):
        piece, n = ndimage.label(lab == k)
        for c in range(1, n + 1):
            out += _split_large(piece == c, max_px, rng, depth + 1, importance, detail_bias)
    return out


def _fragments_dual(subject, face, spacing_bg, spacing_face, min_px, max_px, rng,
                    importance=None, detail_bias=0.0):
    """DUAL-DENSITY tiling: run one Voronoi at `spacing_face` on the FACE region and another
    at `spacing_bg` on the rest, then concatenate. This is a genuinely stronger concentration
    than the semantic-mask importance blend (which only shrinks split-caps ~detail_bias-fold):
    here the face gets its own dense SEEDING, so a small fixed budget can be pushed hard onto
    the face (e.g. ~2/3 of the shards on ~8% of canvas) -- giving the face a LOCAL density
    comparable to a much larger whole-image count. `face` is a boolean mask (subject & ROI)."""
    face_reg = subject & face
    bg_reg = subject & (~face)
    frags = []
    if face_reg.any():
        # allow tiny face shards: floor min_px well below the global floor on the face
        frags += _fragments(face_reg, spacing_face, max(4.0, min_px * 0.15), max_px, rng,
                            importance, detail_bias)
    if bg_reg.any():
        frags += _fragments(bg_reg, spacing_bg, min_px, max_px, rng, importance, detail_bias)
    return frags


def _fragments_regions(subject, labels, spacing_px, min_px, max_px, rng,
                       importance=None, detail_bias=0.0, region_scale=None):
    """Fragment each labelled REGION of `subject` on its OWN Voronoi, then concatenate, so no
    shard ever crosses a region boundary -- the region outlines (e.g. the lemons vs the dish)
    stay crisp instead of being blurred by a straddling shard that can only be one colour.

    `labels` is an int array aligned to `subject`: 0 = unlabelled (fragmented together as a
    remainder), 1..K = regions each tiled independently. Same tiling contract as `_fragments`
    (union of the returned masks == `subject`), just partitioned by region.

    `region_scale` (optional {label: multiplier}) sets each region's own shard spacing:
    <1 packs SMALLER shards into that region, so a face can carry finer detail than a coat
    while both still respect their outlines. The area floor scales with it (a denser region is
    allowed correspondingly smaller pieces); `max_px` stays global so nothing gets huge.
    """
    labels = np.asarray(labels)
    frags = []
    covered = np.zeros_like(subject)
    for v in (int(u) for u in np.unique(labels) if u != 0):
        reg = subject & (labels == v)
        if reg.any():
            sc = float(region_scale.get(v, 1.0)) if region_scale else 1.0
            frags += _fragments(reg, spacing_px * sc, max(4.0, min_px * sc * sc), max_px,
                                rng, importance, detail_bias)
            covered |= reg
    remainder = subject & (~covered)                  # subject pixels outside every region
    if remainder.any():
        frags += _fragments(remainder, spacing_px, min_px, max_px, rng, importance, detail_bias)
    return frags


def _fragments(mask, spacing_px, min_px, max_px, rng, importance=None, detail_bias=0.0):
    """Return a list of boolean fragment masks that FULLY tile `mask`.

    Sub-minimum slivers are not dropped (that would leave holes); instead every silhouette
    pixel is grown into its nearest kept fragment so the union == the silhouette.

    `importance` (optional, per-pixel array matching `mask`, higher = more visual detail
    there) concentrates extra shards in detailed regions instead of spreading the shard
    budget uniformly -- see `_local_cap`. Omit it (default) for the original, spatially-
    uniform behaviour used by the mono/weave paths.
    """
    seeds = _seed_points(mask, spacing_px, rng)
    lab = _voronoi_labels(mask, seeds)
    frag_id = np.zeros(mask.shape, dtype=int)         # 0 = unassigned
    nid = 0
    for k in range(len(seeds)):
        comp, n = ndimage.label(lab == k)
        for c in range(1, n + 1):
            m = comp == c
            a = int(m.sum())
            if a < min_px:
                continue                              # leave for gap-fill (merge)
            cap = _local_cap(m, max_px, importance, detail_bias)
            for sub in (_split_large(m, max_px, rng, importance=importance, detail_bias=detail_bias)
                       if a > cap else [m]):
                nid += 1
                frag_id[sub] = nid
    if nid == 0:
        return []
    # grow kept fragments into the leftover slivers so coverage stays complete
    holes = mask & (frag_id == 0)
    if holes.any():
        inds = ndimage.distance_transform_edt(frag_id == 0, return_indices=True,
                                              return_distances=False)
        nearest = frag_id[tuple(inds)]
        frag_id = np.where(holes, nearest, frag_id)
    return [frag_id == i for i in range(1, nid + 1) if (frag_id == i).any()]


def _importance_map(rgb, semantic=None, semantic_weight=0.0):
    """Per-pixel visual-detail weight, steering where small shards concentrate.

    Base term (`semantic_weight`=0, the control): gradient magnitude of luma -- fires at
    edges/colour transitions. On flat graphics this cleanly marks the interesting bits; on
    an OIL PAINTING it is semantically blind -- brushstroke texture is high-frequency
    everywhere, so the gradient fires across the whole canvas and detail spreads thin
    instead of concentrating on the face.

    `semantic` (optional, per-pixel [0,1] array, 1=want-detail e.g. the face): blended with
    the gradient as `(1-w)*gradient + w*semantic` (w=`semantic_weight`). w=0 reproduces the
    gradient-only behaviour exactly; w=1 ignores the gradient and puts small shards only
    where `semantic` is high. This is the lever for concentrating shards on the face of an
    oil painting rather than smearing them across brushstroke texture."""
    luma = rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140
    g = np.hypot(ndimage.sobel(luma, axis=0), ndimage.sobel(luma, axis=1))
    p95 = np.percentile(g, 95) or 1.0
    grad = np.clip(g / p95, 0.0, 1.0)
    if semantic is not None and semantic_weight > 0.0:
        sem = np.clip(np.asarray(semantic, float), 0.0, 1.0)
        blended = (1.0 - semantic_weight) * grad + semantic_weight * sem
        return blended * 0.85 + 0.15
    return grad * 0.85 + 0.15


def outline_map(rgb, mask, edge_pct=90.0, boundary_px=2, guard_px=3):
    """Per-pixel outline weight in [0,1] -- high on the image's DEFINING contour, tapering off.

    The Scream (and any silhouette-driven image) reads by its outline: the outer silhouette plus
    the strong internal contour (the dark wavy skull-face, the mouth, the hands). Two uses, both
    fed from this one map:
      * as a `semantic` mask for `_importance_map` (concentrate finer shards ON the outline so its
        own shards trace the contour instead of straddling it), and
      * as a `protect` map for `_shard_damage` (make the OTHER wall's stray shadows expensive here,
        so cross-talk is steered off the contour instead of piling onto it -- the damage scorer is
        otherwise edge-blind and treats the dark contour as free space).

    Construction (all inside `mask`, 0 outside):
      1. internal contour band: luma-gradient pixels above the `edge_pct`th percentile,
      2. silhouette boundary band: `mask XOR erosion(mask)`, `boundary_px` wide,
      3. union, dilate `guard_px` into a guard band, and taper by a normalised distance transform
         so weight is strongest on the line and fades outward across the band.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    luma = rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140
    g = np.hypot(ndimage.sobel(luma, axis=0), ndimage.sobel(luma, axis=1))
    thr = np.percentile(g[mask], edge_pct) if mask.any() else 0.0
    contour = mask & (g >= thr)                                    # strong internal edges
    boundary = mask & ~ndimage.binary_erosion(mask, iterations=max(1, boundary_px))
    band = contour | boundary
    if guard_px > 0:
        band = mask & ndimage.binary_dilation(band, iterations=guard_px)
    if not band.any():
        return np.zeros(mask.shape, np.float32)
    # taper: 1 on the seed line, fading to 0 at the edge of the guard band
    dist = ndimage.distance_transform_edt(band)
    w = np.zeros(mask.shape, np.float32)
    d = float(dist.max()) or 1.0
    w[band] = (dist[band] / d).astype(np.float32)
    return w


def _autotune_regions(mask, labels, base_spacing, min_px, max_px, rng, importance, detail_bias,
                      budget, region_scale=None, tol=0.15, max_iter=6):
    """`_autotune_spacing` for the REGION-CONSTRAINED path: same coarsen-until-under-budget
    bisection, but scaling every region's spacing by a common factor so the region partition is
    preserved while the total shard count still respects the fabrication ceiling.

    Without this, region runs silently carry more shards than uniform ones (each region costs at
    least one shard plus border slivers), which by this module's own measurement *lowers*
    SSIM/edge-fidelity -- so a region-vs-uniform comparison would be measuring shard inflation
    rather than the segmentation. Returns (fragments, multiplier, count)."""
    def build(mult):
        scaled = None if region_scale is None else {k: v * mult for k, v in region_scale.items()}
        return _fragments_regions(mask, labels, base_spacing * mult, min_px, max_px, rng,
                                  importance, detail_bias, region_scale=scaled)

    frags = build(1.0)
    n = len(frags)
    if not budget or budget <= 0 or n <= budget:
        return frags, 1.0, n
    lo, hi = 1.0, 4.0
    best = (frags, 1.0, n)
    for _ in range(max_iter):
        mid = (lo * hi) ** 0.5
        frags = build(mid)
        n = len(frags)
        if abs(n - budget) < abs(best[2] - budget):
            best = (frags, mid, n)
        if n <= budget:
            hi = mid
            if n >= budget * (1 - tol):
                break
        else:
            lo = mid
    return best


def _autotune_spacing(mask, base_spacing, min_px, max_px, rng, importance, detail_bias,
                      budget, tol=0.15, max_iter=5):
    """`budget` is a fabrication CEILING, not a target to inflate toward: more shard regions
    consistently measures as *lower* wall-reconstruction fidelity (SSIM/edge-fidelity) even
    though average colour error improves slightly -- each extra region is another
    independent "one dominant colour + one randomly-chosen host panel" decision, and that
    adds boundary noise faster than it resolves real detail. So the natural, detail-biased
    fragment count is used as-is whenever it's already under budget; only coarsen (search for
    a bigger spacing) if it would otherwise overshoot. Returns (fragments, multiplier, count).
    """
    frags = _fragments(mask, base_spacing, min_px, max_px, rng, importance, detail_bias)
    n = len(frags)
    if not budget or budget <= 0 or n <= budget:
        return frags, 1.0, n
    lo, hi = 1.0, 4.0
    best = (frags, 1.0, n)
    for _ in range(max_iter):
        mid = (lo * hi) ** 0.5
        frags = _fragments(mask, base_spacing * mid, min_px, max_px, rng, importance, detail_bias)
        n = len(frags)
        if abs(n - budget) < abs(best[2] - budget):
            best = (frags, mid, n)
        if n <= budget:
            hi = mid
            if n >= budget * (1 - tol):
                break
        else:
            lo = mid
    return best


# ---------------------------------------------------------------------------
# collision resolution (weave crossings)
# ---------------------------------------------------------------------------
def _u_to_col(u, panel, Wp):
    return int(round((u - panel.u_range[0]) / max(panel.u_size, 1e-9) * Wp - 0.5))


def _resolve_collisions(scene, opacity, trim=True):
    """Where family-A (x=xA) and family-B (y=yB) pieces both hold material at a crossing,
    trim a thin clearance band from family B. Returns the number of collisions handled."""
    panel_idx = {p.name: i for i, p in enumerate(scene.panels)}
    _, Wp = scene.solve.panel_res
    half = max(1, int(round((scene.material_thickness + scene.fab.joint_clearance) /
                            (scene.panels[0].u_size / Wp) / 2)))
    hits = 0
    for c in weave_crossings(scene):
        pa = scene.panels[panel_idx[c.panelA_name]]
        pb = scene.panels[panel_idx[c.panelB_name]]
        ia, ib = panel_idx[c.panelA_name], panel_idx[c.panelB_name]
        ca = _u_to_col(c.uA, pa, Wp)                  # A: local u = y = yB
        cb = _u_to_col(c.uB, pb, Wp)                  # B: local u = x = xA
        if not (0 <= ca < Wp and 0 <= cb < Wp):
            continue
        band_a = opacity[ia][:, max(0, ca - half):ca + half + 1].max(axis=1)
        band_b = opacity[ib][:, max(0, cb - half):cb + half + 1].max(axis=1)
        rows = np.where((band_a > 0.5) & (band_b > 0.5))[0]
        if rows.size:
            hits += 1
            if trim:
                opacity[ib][rows[:, None],
                            np.arange(max(0, cb - half), min(Wp, cb + half + 1))[None, :]] = 0.0
    return hits


def count_collisions(scene, opacity):
    return _resolve_collisions(scene, opacity.copy(), trim=False)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def fragment_shards(scene, table, targets, seed=None):
    """Return (opacity [P,Hp,Wp] hard 0/1, fragments:list[dict], collisions_resolved:int)."""
    sp = scene.solve
    seed = sp.seed if seed is None else seed
    Hp, Wp = sp.panel_res
    op = np.zeros((len(scene.panels), Hp, Wp), dtype=np.float32)
    fragments = []

    for fi, family in enumerate(scene.walls.keys()):
        wall = scene.walls[family]
        target = targets[family]
        Hn, Wn = target.shape
        fam = [(gi, p) for gi, p in enumerate(scene.panels)
              if primary_wall_of(scene, table, p) == family]
        if not fam:
            continue
        rng = np.random.default_rng(seed + 17 * fi)

        px_m = 0.5 * (wall.width / Wn + wall.height / Hn)     # metres per wall pixel
        px_area_mm2 = (wall.width / Wn) * (wall.height / Hn) * 1e6
        spacing = sp.fragment_size / max(px_m, 1e-9)
        min_px = sp.fragment_min_area / (px_m ** 2)
        max_px = sp.fragment_max_area / (px_m ** 2)
        gap_it = int(round(sp.shard_gap / max(px_m, 1e-9)))

        frags = _fragments(target > 0.5, spacing, min_px, max_px, rng)

        # depth-assignment weights: near (low magnification) panels optionally favoured
        mags = np.array([table[(p.name, family)].magnification for _, p in fam])
        mrange = float(mags.max() - mags.min()) or 1.0
        w = 1.0 + sp.depth_bias * (mags.max() - mags) / mrange
        w = w / w.sum()

        panel_wall = {gi: np.zeros((Hn, Wn), np.float32) for gi, _ in fam}
        for m in frags:
            if gap_it > 0:
                e = ndimage.binary_erosion(m, iterations=gap_it)
                if e.any():
                    m = e
            local_k = int(rng.choice(len(fam), p=w))
            gi, panel = fam[local_k]
            panel_wall[gi] = np.maximum(panel_wall[gi], m.astype(np.float32))
            shadow_mm2 = float(m.sum()) * px_area_mm2
            fragments.append({"wall": family, "panel": panel.name,
                              "wall_mm2": shadow_mm2,
                              "phys_mm2": shadow_mm2 / (table[(panel.name, family)].magnification ** 2)})

        # back-project each panel's accumulated wall mask onto the panel (once per panel)
        for gi, panel in fam:
            uv = _panel_pixel_uv(panel, (Hp, Wp))
            ab = H.apply_homography(table[(panel.name, family)].H_pw, uv)
            col = ab[..., 0] / max(wall.width, 1e-9) * Wn - 0.5
            row = ab[..., 1] / max(wall.height, 1e-9) * Hn - 0.5
            s = ndimage.map_coordinates(panel_wall[gi], [row.ravel(), col.ravel()],
                                        order=0, mode="constant", cval=0.0).reshape(Hp, Wp)
            op[gi] = (s > 0.5).astype(np.float32)

    resolved = _resolve_collisions(scene, op, trim=True)
    return op, fragments, resolved


# ---------------------------------------------------------------------------
# CMYK colour: split into C/M/Y/K channels; tone by STACKING same-colour shards
# across depth planes; mixing = different-colour shards on the same ray.
# ---------------------------------------------------------------------------
def _layer_counts(cvals, max_layers, budget):
    """Channel intensities {C,M,Y,K}->0..1 -> integer layer counts, total capped at `budget`."""
    counts = {ch: min(max_layers, int(round(cvals[ch] * max_layers))) for ch in _color.CMYK}
    total = sum(counts.values())
    while total > budget:                                  # drop from the densest channel
        ch = max(counts, key=lambda k: counts[k])
        counts[ch] -= 1
        total -= 1
    return counts


def fragment_shards_cmyk(scene, table, targets, names=None, white_thr=0.90,
                         max_layers=2, seed=None):
    """CMYK decomposition. `targets` are RGB maps [Hn,Wn,3] in [0,1].

    Each chunky shard region is split into C/M/Y/K layers (count per channel ~ that channel's
    intensity, total <= #planes); each layer is placed on a distinct, randomly chosen depth
    plane so same-colour layers STACK on the ray (tone) and channels SCATTER across planes
    (cryptic). Returns (opacity[P,Hp,Wp], fragments[family,panel,channel,phys_mm2],
    colorid[P,Hp,Wp] indexing `names`, collisions_resolved, stack_depths[list per region]).
    """
    sp = scene.solve
    seed = sp.seed if seed is None else seed
    names = names or (["clear"] + _color.CMYK)
    Hp, Wp = sp.panel_res
    colorid = np.zeros((len(scene.panels), Hp, Wp), dtype=np.int16)
    fragments, stack_depths = [], []

    for fi, family in enumerate(scene.walls.keys()):
        wall = scene.walls[family]
        rgb = targets[family]
        Hn, Wn = rgb.shape[:2]
        fam = [(gi, p) for gi, p in enumerate(scene.panels)
              if primary_wall_of(scene, table, p) == family]
        if not fam:
            continue
        n_planes = len(fam)
        rng = np.random.default_rng(seed + 17 * fi)

        px_m = 0.5 * (wall.width / Wn + wall.height / Hn)
        px_area_mm2 = (wall.width / Wn) * (wall.height / Hn) * 1e6
        spacing = sp.fragment_size / max(px_m, 1e-9)
        min_px = sp.fragment_min_area / (px_m ** 2)
        max_px = sp.fragment_max_area / (px_m ** 2)

        frags = _fragments(_color.subject_mask(rgb, white_thr), spacing, min_px, max_px, rng)

        panel_wall_id = {gi: np.zeros((Hn, Wn), np.int16) for gi, _ in fam}
        for m in frags:
            cvals = _color.rgb_to_cmyk(_color.dominant_rgb(rgb[m]))   # dominant colour, not blend
            counts = _layer_counts(cvals, max_layers, n_planes)
            instances = [ch for ch in _color.CMYK for _ in range(counts[ch])]
            if not instances:
                continue
            planes = rng.choice(n_planes, size=len(instances), replace=False)  # distinct planes
            stack_depths.append(len(instances))
            shadow_mm2 = float(m.sum()) * px_area_mm2
            for ch, pl in zip(instances, planes):
                gi, panel = fam[int(pl)]
                panel_wall_id[gi][m] = names.index(ch)
                fragments.append({"wall": family, "panel": panel.name, "channel": ch,
                                  "phys_mm2": shadow_mm2 / table[(panel.name, family)].magnification ** 2})

        for gi, panel in fam:
            uv = _panel_pixel_uv(panel, (Hp, Wp))
            ab = H.apply_homography(table[(panel.name, family)].H_pw, uv)
            col = ab[..., 0] / max(wall.width, 1e-9) * Wn - 0.5
            row = ab[..., 1] / max(wall.height, 1e-9) * Hn - 0.5
            s = ndimage.map_coordinates(panel_wall_id[gi], [row.ravel(), col.ravel()],
                                        order=0, mode="constant", cval=0).reshape(Hp, Wp)
            colorid[gi] = s.astype(np.int16)

    opacity = (colorid > 0).astype(np.float32)
    resolved = _resolve_collisions(scene, opacity, trim=True)
    colorid[opacity == 0] = 0
    return opacity, fragments, colorid, resolved, stack_depths


# ---------------------------------------------------------------------------
# Stochastic Shard Overlap: channels stay on the SAME depth plane(s) (1-2 per
# family) and mix by lamination instead of by scattering onto separate planes.
# ---------------------------------------------------------------------------
def _active_channels(cvals, max_stack, thresh=0.15):
    """Which CMYK channels a region needs, strongest first, capped at `max_stack`.

    A region needing a secondary colour (e.g. violet) lights up two channels above
    `thresh`; a tertiary/dark mix lights up three. Pure regions fall back to their single
    strongest channel so we never fabricate a mix that isn't in the source pixels."""
    ranked = sorted(_color.CMYK, key=lambda ch: cvals[ch], reverse=True)
    active = [ch for ch in ranked if cvals[ch] >= thresh][:max_stack]
    return active or ranked[:1]


def _compromise_channels(dom_P, dom_S, w, names, max_stack):
    """A shard colour that serves BOTH walls: blend the shard's own dominant colour `dom_P`
    toward what the other wall wants at its landing (`dom_S`) by fraction `w`, then push the
    wish through the SAME realisable path the pipeline always uses (CMYK split -> strongest
    <=max_stack channels -> laminated transmittance). Returns (active, cvals, T).

    `w=0` returns exactly `_shard_channels`' result, so the feature is off by default.
    Measured headroom (out_thickness_test/headroom.py): on the secondary wall this roughly
    DOUBLES the reachable colour-agreeing double duty (e.g. 11.0% -> 21.8%) for a primary-wall
    colour cost of ~0.01, because today's colour is chosen without ever consulting wall B."""
    wish = (1.0 - w) * np.asarray(dom_P, float) + w * np.asarray(dom_S, float)
    palette = [n for n in names if n != "clear"]
    if _color.has_cmyk_channels(names):
        cvals = _color.rgb_to_cmyk(wish)
        active = _active_channels(cvals, max_stack)
    else:
        idx = int(_color.quantize_ids(wish[None, :], palette)[0])
        active, cvals = [palette[idx]], {palette[idx]: 1.0}
    return active, cvals, _shard_transmit(cvals, active)


def _secondary_dominant(ys, xs, G, wallP, wallS, target_S, res_P, res_S,
                        rng, max_samples=200):
    """What the OTHER wall wants where this shard would land if hosted on the panel whose
    wall->wall homography is `G`: the median wanted colour over the landing footprint, or None
    if the shard lands off that wall entirely. Same projection as `_shard_damage` (which already
    reads these pixels to SCORE a colour); here we read them to CHOOSE one."""
    n = len(ys)
    if n == 0:
        return None
    if n > max_samples:
        sel = rng.choice(n, size=max_samples, replace=False)
        ys, xs = ys[sel], xs[sel]
    HnP, WnP = res_P
    HnS, WnS = res_S
    a = (xs + 0.5) / WnP * wallP.width
    b = (ys + 0.5) / HnP * wallP.height
    q = H.apply_homography(G, np.stack([a, b], axis=-1))
    ci = np.rint(q[:, 0] / max(wallS.width, 1e-9) * WnS - 0.5).astype(int)
    ri = np.rint(q[:, 1] / max(wallS.height, 1e-9) * HnS - 0.5).astype(int)
    on = (ci >= 0) & (ci < WnS) & (ri >= 0) & (ri < HnS)
    if not on.any():
        return None
    return np.median(target_S[ri[on], ci[on]], axis=0)


def _shard_channels(region_rgb, names, max_stack):
    """Per-shard (active channel list, cvals dict) for the overlap loop, palette-aware.

    CMYK-style palette (has C/M/Y members): the existing CMYK-separation path -- split the
    shard's dominant colour into process channels and laminate the strongest few.

    Non-CMYK palette (e.g. 'noir', pure greys): CMYK separation is meaningless, so snap the
    dominant colour straight to the single nearest palette entry by transmittance
    (`color.quantize_ids`) and lay one full-strength sheet. Tone still comes from the ramp of
    grey stock colours + stacking, not from channel intensities."""
    dom = _color.dominant_rgb(region_rgb)
    palette = [n for n in names if n != "clear"]
    if _color.has_cmyk_channels(names):
        cvals = _color.rgb_to_cmyk(dom)
        return _active_channels(cvals, max_stack), cvals
    # nearest palette colour by transmittance (same measure quantize uses everywhere else)
    idx = int(_color.quantize_ids(dom[None, :], palette)[0])
    ch = palette[idx]
    return [ch], {ch: 1.0}


def _wall_to_wall_H(table, panel, primary, secondary):
    """3x3 homography: a point on the PRIMARY wall -> where the same shard material, if
    hosted on `panel`, casts on the SECONDARY wall.

    Composes the two homographies already in the projection table:
        wall_P (metres) --H_wp[panel,P]--> panel (u,v) --H_pw[panel,S]--> wall_S (metres)
    Collapsing the round-trip into ONE matrix (computed once per panel, outside the shard
    loop) is what makes cross-talk-aware assignment cheap: a shard's damage can then be
    evaluated by transforming only its OWN pixel coordinates, with no wall re-render."""
    return table[(panel.name, secondary)].H_pw @ table[(panel.name, primary)].H_wp


_LUMA_W = np.array([0.2989, 0.5870, 0.1140])          # Rec.601, matches metrics._luma


def _shard_damage(ys, xs, G, wallP, wallS, target_S, absorb, res_P, res_S,
                  rng, max_samples=200, transmit=None, credit_weight=None, agree_min=None,
                  match_tol=0.30, match_metric="rgb", protect=None, protect_weight=0.0):
    """Visual error this shard would inflict on the SECONDARY wall if hosted on the panel
    whose wall->wall homography is `G`. Lower is better; 0 = harmless.

        D = mean over the shard's pixels of || target_S(q) * (1 - T_shard) ||^2

    where q is where that pixel's material lands on the secondary wall and `absorb` =
    (1 - T_shard) is how much light the shard removes per RGB channel. The shard turns
    target_S(q) into target_S(q)*T_shard, so target_S(q)*(1-T_shard) is exactly the light
    it steals that the secondary image wanted to keep. This gives the right behaviour for
    free:
      - secondary target is DARK there -> ~0 damage (a stray shadow on an already-dark
        region is harmless -- the on-subject/off-subject split used earlier got this wrong),
      - secondary target is WHITE background -> maximal damage (the visible stray-shard case),
      - colour mismatch is penalised per channel (a red shard over blue steals the blue),
      - lands OFF the wall entirely -> contributes zero (so "steer the cross-talk off the
        wall" is a strategy the assignment can discover by itself).

    Samples at most `max_samples` of the shard's own pixels -- no wall is rendered."""
    n = len(ys)
    if n == 0:
        return 0.0
    if n > max_samples:                                   # subsample: damage is a mean, so a
        sel = rng.choice(n, size=max_samples, replace=False)   # few hundred px is plenty
        ys, xs = ys[sel], xs[sel]
    HnP, WnP = res_P
    HnS, WnS = res_S
    a = (xs + 0.5) / WnP * wallP.width                    # primary-wall pixel -> metres
    b = (ys + 0.5) / HnP * wallP.height
    q = H.apply_homography(G, np.stack([a, b], axis=-1))  # -> secondary-wall metres
    col = q[:, 0] / max(wallS.width, 1e-9) * WnS - 0.5    # -> secondary-wall pixel
    row = q[:, 1] / max(wallS.height, 1e-9) * HnS - 0.5
    ci = np.rint(col).astype(int)
    ri = np.rint(row).astype(int)
    on = (ci >= 0) & (ci < WnS) & (ri >= 0) & (ri < HnS)  # off-wall -> harmless -> 0 damage
    if not on.any():
        return 0.0
    tgt = target_S[ri[on], ci[on]]                        # [k,3] secondary target there

    # OUTLINE PROTECTION (`protect`, a [Hn_S,Wn_S] weight in [0,1] high on the secondary wall's
    # defining contour): the harm terms below are edge-blind -- a stray shadow on a DARK contour
    # steals ~no light so it reads as harmless (and the signed credit can even reward it), which
    # is exactly what lets cross-talk pile onto the very outline that carries the image. `w` is
    # that contour weight at each landing pixel; where it is high we make landing expensive
    # (amplify harm, suppress credit) so the host greedy steers the stray shadow off the outline.
    pw = protect_weight if (protect is not None and protect_weight) else 0.0
    w = protect[ri[on], ci[on]] if pw else None

    if credit_weight is None:                             # UNSIGNED (harm-only, >= 0)
        stolen = tgt * absorb[None, :]                    # light removed that was wanted
        per_px = (stolen ** 2).sum(axis=1)                # off-wall pixels count as 0
        if w is not None:                                 # landing on the outline costs more
            per_px = per_px * (1.0 + pw * w)
        return float(per_px.sum()) / len(ci)

    # SIGNED: damage = error_with - error_without, allowed to go NEGATIVE (a credit).
    # Without the shard the secondary pixel is unblocked white (1,1,1); with it, the light
    # through is the shard's transmittance T. So:
    #     e_without = ||1 - target||^2      (how wrong "leave it blank" already is)
    #     e_with    = ||T - target||^2      (how wrong it is once this shard shadows it)
    # target WHITE  -> e_without=0, any shadow is pure harm            -> positive
    # target BLACK  -> white is maximally wrong, a dark shard fixes it -> negative (credit)
    # target ~ T    -> the shard supplies the very colour wanted       -> negative (credit)
    # This is what lets a shard be REWARDED for doing genuine double duty, which the
    # harm-only form structurally could not do (it drove good cross-talk to zero along with
    # bad, because helping was never worth anything). `credit_weight` scales ONLY the
    # negative branch, so the credit/penalty balance can be swept independently.
    e_without = ((1.0 - tgt) ** 2).sum(axis=1)
    e_with = ((transmit[None, :] - tgt) ** 2).sum(axis=1)
    signed = e_with - e_without

    # COLOUR-AGREEMENT credit (option (a)). The whole reason a credit term is needed is that
    # a shard's colour and shape are both fixed on its OWN wall BEFORE assignment; assignment
    # only picks depth, i.e. WHERE the stray shadow lands, never WHAT colour arrives. So the
    # stray shadow always carries the primary wall's colour, chosen without ever consulting
    # the secondary image. The credit must therefore reward only GENUINE contribution: the
    # stray shadow supplying the colour the secondary image actually wants there.
    #
    # Two earlier forms both over-credited contamination:
    #   * raw `e_with - e_without` (distance-from-WHITE) rewarded ANY dark shard on a dark
    #     target for merely being "darker than white" -- a red shard on the golden croissant
    #     scored as helpful.
    #   * the hue-only chroma cosine gate ignored brightness and was meaningless on near-
    #     neutral (grey/dark) targets, where chroma is undefined.
    # Measured: under either, "B_good" read ~27% while the honest colour-agreeing fraction was
    # ~3% -- it was counting "landed on content", not "supplied the right colour".
    #
    # (a): a stray pixel earns credit ONLY where the transmitted colour genuinely matches the
    # secondary target there in FULL RGB, ||T - target|| < `match_tol`. This is the identical
    # test used to report colour-agreeing B_good, so the score the optimiser chases and the
    # metric we report are the same quantity. A stray pixel that improves on blank but with the
    # wrong colour earns nothing (neither credit nor extra harm); genuine harm (signed>0) is
    # left untouched. `match_tol=None` restores the legacy hue-gate for comparison only.
    if match_tol is not None:
        if match_metric == "lab":
            # perceptual gate in CIELAB (CIE76 dE): "right colour" is judged the way the eye sees
            # colour difference rather than by raw-RGB Euclidean distance, so near-neutral and dark
            # targets (where RGB distance is misleading) are handled correctly and fewer wrong-
            # colour specks earn the credit. `match_tol` is then a dE (~15-25), not an RGB distance.
            dist = _color.delta_e(transmit[None, :], tgt)
        elif match_metric == "luma":
            # perceptual gate: weight the per-channel error by Rec.601 luma before measuring
            # "right colour", so a mismatch the eye barely notices (e.g. in blue) is not
            # counted as harshly as one it does (green). Off by default (match_metric="rgb").
            diff = (transmit[None, :] - tgt) * _LUMA_W[None, :]
            dist = np.sqrt((diff ** 2).sum(axis=1))
        else:
            dist = np.sqrt(((transmit[None, :] - tgt) ** 2).sum(axis=1))   # full-RGB, brightness incl.
        match = dist < match_tol
        signed = np.where((signed < 0.0) & match, credit_weight * signed,
                          np.where(signed < 0.0, 0.0, signed))
    elif agree_min is not None and agree_min > 0.0:                    # legacy hue gate (retired)
        nt = transmit / (np.linalg.norm(transmit) + 1e-6)
        ng = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-6)
        agree = ng @ nt
        gate = np.clip((agree - agree_min) / max(1.0 - agree_min, 1e-6), 0.0, 1.0)
        signed = np.where(signed < 0.0, credit_weight * gate * signed, signed)
    else:
        signed = np.where(signed < 0.0, credit_weight * signed, signed)
    if w is not None:                                      # outline guard: amplify harm on the
        signed = np.where(signed > 0.0, signed * (1.0 + pw * w),         # contour, and never let
                          signed * (1.0 - np.clip(w, 0.0, 1.0)))         # it earn a dark-on-dark credit
    return float(signed.sum()) / len(ci)                   # off-wall pixels count as 0 (neutral)


def _shard_transmit(cvals, active):
    """Laminated transmittance of one shard's CMYK stack, intensity-weighted -- the SAME
    model the renderer uses (color.stack_transmit_lut), so the damage estimate matches what
    will actually be rendered rather than a separate approximation."""
    T = np.ones(3, dtype=float)
    for ch in active:
        T *= 1.0 - cvals[ch] * (1.0 - _color.transmit_rgb(ch))
    return T


def _panel_reachable_mask(panel, H_pw, wall, Hn, Wn):
    """Boolean [Hn,Wn]: which wall pixels this panel can actually cast material onto --
    i.e. whose back-projected (u,v) (through the inverse of the panel->wall homography)
    falls inside the panel's own `u_range`/`v_range`, not just "primary to this wall".

    This matters once panels stop being full-wall-width strips (see panel_search.py):
    a small, arbitrarily positioned/angled panel's projected footprint may only cover a
    fraction of the wall, or none of a given region at all. Assigning a shard to it
    uniformly at random (the old behaviour, safe only because every fixed-layout panel
    spanned the whole wall) silently drops that shard's material wherever the panel can't
    reach -- confirmed empirically: 0% survival on several search-generated panels."""
    try:
        H_wp = np.linalg.inv(H_pw)
    except np.linalg.LinAlgError:
        return np.zeros((Hn, Wn), dtype=bool)
    rows, cols = np.mgrid[0:Hn, 0:Wn]
    a = (cols.ravel() + 0.5) / Wn * wall.width
    b = (rows.ravel() + 0.5) / Hn * wall.height
    uv = H.apply_homography(H_wp, np.stack([a, b], axis=-1))
    u0, u1 = panel.u_range
    v0, v1 = panel.v_range
    ok = (uv[:, 0] >= u0) & (uv[:, 0] <= u1) & (uv[:, 1] >= v0) & (uv[:, 1] <= v1)
    return ok.reshape(Hn, Wn)


def fragment_shards_overlap(scene, table, targets, names=None, white_thr=0.90,
                            max_stack=None, jitter=None, seed=None,
                            shard_budget=None, detail_bias=None,
                            penumbra_min_feature_sigmas=None, damage_weight=0.0,
                            credit_weight=None, agree_min=None, match_tol=0.30,
                            match_metric="rgb", diagonal_frac=0.0,
                            semantic_masks=None, semantic_weight=0.0,
                            face_masks=None, face_density=1.0, bg_coarsen=1.0,
                            region_masks=None, region_scales=None,
                            colour_blend=0.0, colour_primary_tol=0.10,
                            outline_masks=None, outline_protect_weight=0.0,
                            intensity_gain=1.0, joint_prior=None, joint_weight=1.0):
    """'Stochastic Shard Overlap': interleave C/M/Y/K shards onto the SAME set of depth
    planes (typically 1-2 per family) instead of separating each channel onto its own plane.

    Each organic shard region gets up to `max_stack` channel layers laminated at that same
    (u,v) footprint, each at a slightly different micro depth-offset within the panel's
    thickness (see shard_geom.build_stack_mesh) so secondary/tertiary colours mix
    subtractively right there. Each layer's footprint is also given a small independent
    random jitter (driven by `jitter` x shard size) so adjacent colour regions no longer
    share a razor-edge boundary that traces the source image exactly. The whole stack is
    also scattered onto a randomly chosen panel in its family, so the result reads as a
    chaotic cloud of overlapping brushstrokes rather than a tidy per-channel tiling.

    Shard placement is detail-adaptive rather than spatially uniform: `detail_bias`
    concentrates shards where the source image actually has edges/colour transitions (see
    `_importance_map`/`_local_cap`) instead of spreading a fixed budget evenly -- measured to
    give a small fidelity gain on its own. Deliberately inflating the *count* of shards
    further does not: more (smaller) regions consistently score *lower* on SSIM/edge-
    fidelity even as colour error (RMSE) improves slightly, because each extra region is
    another independent "one dominant colour + one randomly-chosen host panel" decision, and
    that adds boundary noise faster than it resolves real detail. So `shard_budget` is a
    fabrication CEILING, not a target: the natural, detail-biased fragment count is used as
    long as it's already under budget, and `_autotune_spacing` only coarsens (fewer, bigger
    shards) if it would otherwise be exceeded -- keeping the shard count doable without
    trading away fidelity to get there.

    Each fragment also records `softening_sigmas` = how many multiples of that panel's own
    physical penumbra radius the jitter amounts to (the renderer already Gaussian-blurs each
    panel's contribution by that radius before compositing). A ratio near/below 1 means the
    scatter is essentially free (absorbed by the existing blur); much larger than 1 means
    the cast shadow is being deliberately softened to buy more scrambling -- surfaced so
    that tradeoff is visible per run (see the CLI summary) rather than silently capped or
    silently ignored.

    `penumbra_min_feature_sigmas` makes the fragmentation itself penumbra-aware (as opposed
    to `softening_sigmas` above, which only *reports* the jitter/penumbra relationship): it
    raises the minimum feature size fed to `_fragments` to this many multiples of the
    worst-case (farthest) panel's physical penumbra sigma in that family, so shard detail
    finer than what a real, non-ideal (non-point) light could ever resolve gets merged into
    its neighbour by the existing gap-fill path instead of being fabricated as illusory
    precision. 0 disables it (restores the previous `fragment_min_area`-only floor exactly).

    `outline_masks` ({family: [Hn,Wn] weight in [0,1], high on that wall's defining contour --
    build with `outline_map`) + `outline_protect_weight` make the host greedy AVOID landing a
    family's stray cross-talk on the OTHER wall's outline: `_shard_damage` is edge-blind (a
    shadow on a dark contour steals ~no light, so it scores harmless and can even earn a credit),
    which is exactly what lets cross-talk pile onto the line that carries the image. The weight
    amplifies harm and suppresses credit there. Pair with `semantic_masks=outline_masks`,
    `semantic_weight>0` to also concentrate finer shards ON the outline so its own shards trace
    it instead of straddling. `outline_protect_weight=0`/`outline_masks=None` -> old behaviour.

    `intensity_gain` (>1, opt-in, default 1.0 = off) lifts every placed channel's recorded intensity
    (clipped at 1.0) so deep colours render as dark/saturated as the physical full sheet actually
    casts, instead of the intensity-weighted render washing Prussian blue to mid-blue -- see the
    inline note at the stacking loop. Hue is preserved (weak channels stay proportionally weakest).

    Returns (stack_colorid [S,P,Hp,Wp] int16, opacity [P,Hp,Wp] float32, fragments:list[dict],
    collisions_resolved:int, stack_depths:list[int], budget_stats:dict[family -> {target,
    achieved, spacing_multiplier, penumbra_min_feature_mm}], stack_intensity [S,P,Hp,Wp]
    float32 -- per-slot CMYK channel intensity for intensity-weighted lamination, feed to
    color.stack_transmit_lut).
    """
    sp = scene.solve
    seed = sp.seed if seed is None else seed
    names = names or (["clear"] + _color.CMYK)
    max_stack = scene.color_max_stack if max_stack is None else max_stack
    jitter = scene.overlap_jitter if jitter is None else jitter
    shard_budget = scene.overlap_shard_budget if shard_budget is None else shard_budget
    detail_bias = scene.overlap_detail_bias if detail_bias is None else detail_bias
    penumbra_min_feature_sigmas = (scene.overlap_penumbra_min_feature_sigmas
                                   if penumbra_min_feature_sigmas is None
                                   else penumbra_min_feature_sigmas)
    Hp, Wp = sp.panel_res
    S = max(1, int(max_stack))
    stack_colorid = np.zeros((S, len(scene.panels), Hp, Wp), dtype=np.int16)
    stack_intensity = np.zeros((S, len(scene.panels), Hp, Wp), dtype=np.float32)
    fragments, stack_depths = [], []
    budget_stats = {}

    for fi, family in enumerate(scene.walls.keys()):
        wall = scene.walls[family]
        rgb = targets[family]
        Hn, Wn = rgb.shape[:2]
        fam = [(gi, p) for gi, p in enumerate(scene.panels)
              if primary_wall_of(scene, table, p) == family]
        if not fam:
            continue
        n_planes = len(fam)
        rng = np.random.default_rng(seed + 17 * fi)

        px_m = 0.5 * (wall.width / Wn + wall.height / Hn)
        px_area_mm2 = (wall.width / Wn) * (wall.height / Hn) * 1e6
        spacing = sp.fragment_size / max(px_m, 1e-9)
        min_px = sp.fragment_min_area / (px_m ** 2)
        max_px = sp.fragment_max_area / (px_m ** 2)

        # penumbra-aware minimum feature size: a real (non-ideal) light of finite size
        # blurs each panel's contribution by that panel's own penumbra_sigma_m before it
        # ever reaches the wall (see geometry/projection.py, forward/renderer.py's blur
        # pass) -- shard detail finer than that is fabricated precision the light can never
        # actually resolve. Use the WORST-case (largest, i.e. farthest) panel in the family
        # so the floor holds regardless of which panel a region ends up scattered onto.
        penumbra_floor_px = max((table[(p.name, family)].penumbra_sigma_m / max(px_m, 1e-9)
                                for _, p in fam), default=0.0)
        min_feature_px = penumbra_min_feature_sigmas * penumbra_floor_px
        min_px_eff = max(min_px, min_feature_px ** 2)

        sem = None if semantic_masks is None else semantic_masks.get(family)
        importance = _importance_map(rgb, semantic=sem, semantic_weight=semantic_weight)
        subj_mask = _color.subject_mask(rgb, white_thr)
        face = None if face_masks is None else (np.asarray(face_masks.get(family)) > 0.5)
        region = None if region_masks is None else region_masks.get(family)
        if region is not None:
            # REGION-CONSTRAINED: fragment each labelled region (e.g. the lemons, the dish)
            # on its own Voronoi so no shard crosses a region boundary -- keeps the outlines
            # accurate.
            #
            # The budget MUST still be honoured here. Tiling each region independently inflates
            # the shard count (every region costs at least one shard, plus slivers along its
            # border), and `_autotune_spacing`'s docstring records that more shard regions
            # measure as *lower* SSIM/edge-fidelity. Leaving this unbudgeted made region runs
            # look worse than uniform ones for a reason that had nothing to do with the
            # segmentation -- they were simply carrying ~15% more shards.
            frags, mult, n_frags = _autotune_regions(
                subj_mask, region, spacing, min_px_eff, max_px, rng, importance, detail_bias,
                shard_budget, region_scale=(region_scales or {}).get(family))
        elif face is not None and face_density > 1.0:
            # DUAL-DENSITY: dense Voronoi on the face, coarse on the rest. Strong
            # concentration -- unlike the (null) semantic mask, the face gets its own
            # dense seeding so a fixed budget can be pushed hard onto the face.
            frags = _fragments_dual(subj_mask, face, spacing * bg_coarsen,
                                    spacing / np.sqrt(face_density), min_px_eff, max_px, rng,
                                    importance, detail_bias)
            mult, n_frags = 1.0, len(frags)
        else:
            frags, mult, n_frags = _autotune_spacing(
                subj_mask, spacing, min_px_eff, max_px, rng,
                importance, detail_bias, shard_budget)
        budget_stats[family] = {"target": shard_budget, "achieved": n_frags,
                                "spacing_multiplier": mult,
                                "penumbra_min_feature_mm": min_feature_px * px_m * 1000.0}

        # which wall pixels each panel in this family can actually reach (see
        # _panel_reachable_mask) -- used below so shard->panel assignment is
        # coverage-aware instead of assuming every panel spans the whole wall.
        reach = {gi: _panel_reachable_mask(panel, table[(panel.name, family)].H_pw, wall, Hn, Wn)
                for gi, panel in fam}

        # --- cross-talk-aware host selection -------------------------------------------
        # Which panel hosts a shard IS the shard's depth, and depth is the ONLY free
        # parameter steering where its unavoidable secondary-wall shadow lands (its primary
        # landing is fixed by the homography regardless of host). That lever used to be a
        # coin flip (`rng.choice`) -- which is why bad cross-talk swung 14-37% purely with
        # the seed. Here it is chosen to minimise the damage that stray shadow does to the
        # OTHER wall's image, using `_shard_damage` (no re-render; see `_wall_to_wall_H`).
        wall_names = list(scene.walls.keys())
        # ALL other walls this family's stray shadow can fall on. A panel casts a shadow on EVERY
        # wall it isn't parallel to, so with 3+ walls each host does cross-talk damage to more than
        # one image at once. Scoring only ONE cyclic-neighbour "secondary" (as before) left every
        # third-wall setup half-blind -- e.g. with [A,B,C], family B's stray shadows landing on A
        # were never counted, so A's image could be wrecked by B with the solver unaware. Summing
        # `_shard_damage` over all others fixes that. For 2 walls `others` is the single other wall,
        # so the sum has one term and this is byte-identical to the previous behaviour.
        others = [w for w in wall_names if w != family]
        wallS_by = {o: scene.walls[o] for o in others}
        targetS_by = {o: targets[o] for o in others}
        resS_by = {o: targets[o].shape[:2] for o in others}
        # outline of each OTHER wall (where a stray shadow from this family would land): the damage
        # calls below make landing on it expensive, so this family's cross-talk is steered off the
        # other images' defining contours. None/0 weight -> identical to the old scorer.
        protectS_by = {o: (None if outline_masks is None else outline_masks.get(o)) for o in others}
        Gs_by = {o: {gi: _wall_to_wall_H(table, p, family, o) for gi, p in fam} for o in others}
        # single-secondary aliases for the colour-blend branch below (a 2-wall colour feature; with
        # one other wall these ARE that wall). colour_blend is NOT generalised to N walls -- it
        # compromises a shard's colour toward one other image, which has no unambiguous 3-wall
        # meaning; no current caller uses colour_blend with a third wall.
        secondary = others[0]
        wallS, target_S, res_S = wallS_by[secondary], targetS_by[secondary], resS_by[secondary]
        protect_S, Gs = protectS_by[secondary], Gs_by[secondary]

        # JOINT PRIOR (B2): `joint_prior[family]` is [P,Hn,Wn] -- each panel's opacity from the
        # two-wall JOINT optimiser (optimizer.solve), warped to THIS wall. A high value at a shard's
        # location on panel gi means the joint solution (which satisfies BOTH walls at once) placed
        # material there, so hosting the shard on gi inherits that joint, cross-talk-exploiting depth
        # choice rather than the per-family "dodge the other wall" heuristic. None -> no effect.
        jpf = None if joint_prior is None else joint_prior.get(family)

        panel_wall_stack = {gi: np.zeros((S, Hn, Wn), np.int16) for gi, _ in fam}
        panel_wall_intensity = {gi: np.zeros((S, Hn, Wn), np.float32) for gi, _ in fam}

        # DIAGONAL QUOTA: the greedy always prefers wall-parallel panels (a diagonal casts a
        # more keystoned shadow -> higher error), so left alone it puts NOTHING on the
        # diagonals and the piece becomes trivial (each wall's image lives on one readable
        # flat plane). `diagonal_frac` forces that share of this family's shards onto the
        # diagonal planes -- among the diagonals the host is still chosen by the same
        # coverage/cross-talk objective, so cross-talk stays minimised, just within the
        # diagonal subset. Which shards are forced is chosen deterministically (evenly spaced
        # by index) so it's reproducible and spatially spread, not clustered.
        diag_local = [k for k, (_, p) in enumerate(fam) if p.is_diagonal()]
        n_force = 0
        force_set = set()
        if diagonal_frac > 0.0 and diag_local:
            n_force = int(round(diagonal_frac * len(frags)))
            if n_force > 0:
                step = len(frags) / n_force
                force_set = {int(j * step) for j in range(n_force)}

        for fidx, m in enumerate(frags):
            active, cvals = _shard_channels(rgb[m], names, S)
            ys, xs = np.where(m)
            cover = np.array([reach[gi][ys, xs].mean() for gi, _ in fam])
            viable = np.flatnonzero(cover > 0)             # ONLY panels that can actually host it
            if fidx in force_set:
                # restrict to diagonal planes this shard can actually reach; if none reach it,
                # fall back to the normal viable set rather than dropping the shard.
                diag_viable = np.array([k for k in diag_local if k in set(viable.tolist())], dtype=int)
                if diag_viable.size:
                    viable = diag_viable
            if viable.size == 0:
                local_k = int(rng.integers(n_planes))      # no panel reaches this shard at all
            elif damage_weight <= 0.0:
                # legacy behaviour (control arm): random, weighted by primary-wall coverage
                # -- restricted to `viable` so a diagonal-forced shard (or an off-wall panel)
                # is honoured here too, not just in the damage-aware branch below.
                cv = cover[viable]
                p = cv / cv.sum() if cv.sum() > 0 else None
                local_k = int(rng.choice(viable, p=p))
            elif colour_blend > 0.0:
                # JOINT (depth + colour): today's colour is fixed from THIS wall before the host
                # is picked, so a shard can only help the other wall by luck. Here the colour is
                # a FUNCTION of the candidate host -- for each panel we look at what the other
                # wall wants where the shard would land and blend toward it -- and the
                # (panel, colour) pair is chosen together. Voronoi (the shard's SHAPE) is
                # untouched; measured straddle ~0.95 says shape is not the limiter.
                dom_P = _color.dominant_rgb(rgb[m])
                err_a = float(np.sqrt(((_shard_transmit(cvals, active) - dom_P) ** 2).sum()))
                best_obj, best_k, best_col = -np.inf, int(viable[0]), (active, cvals)
                for k in viable:
                    G = Gs[fam[k][0]]
                    dom_S = _secondary_dominant(ys, xs, G, wall, wallS, target_S,
                                                (Hn, Wn), res_S, rng)
                    act_k, cv_k = active, cvals                 # fall back to own-wall colour
                    if dom_S is not None:
                        a2, c2, T2 = _compromise_channels(dom_P, dom_S, colour_blend, names, S)
                        # only accept a compromise that stays close enough to what THIS wall
                        # wants -- the primary wall is what a viewer mostly reads.
                        if float(np.sqrt(((T2 - dom_P) ** 2).sum())) - err_a <= colour_primary_tol:
                            act_k, cv_k = a2, c2
                    T_k = _shard_transmit(cv_k, act_k)
                    d_k = _shard_damage(ys, xs, G, wall, wallS, target_S, 1.0 - T_k,
                                        (Hn, Wn), res_S, rng, transmit=T_k,
                                        credit_weight=credit_weight, agree_min=agree_min,
                                        match_tol=match_tol, match_metric=match_metric,
                                        protect=protect_S, protect_weight=outline_protect_weight)
                    o_k = cover[k] - damage_weight * d_k
                    if jpf is not None:
                        o_k = o_k + joint_weight * float(jpf[fam[k][0]][ys, xs].mean())
                    if o_k > best_obj:
                        best_obj, best_k, best_col = o_k, int(k), (act_k, cv_k)
                local_k = best_k
                active, cvals = best_col
            else:
                T_shard = _shard_transmit(cvals, active)
                absorb = 1.0 - T_shard
                # sum the harm this shard would do across ALL other walls (each _shard_damage
                # returns 0 where the shadow lands off that wall, so this naturally accounts for
                # whichever walls a given host actually casts onto). 2 walls -> one term -> as before.
                dmg = np.array([
                    sum(_shard_damage(ys, xs, Gs_by[o][fam[k][0]], wall, wallS_by[o], targetS_by[o],
                                      absorb, (Hn, Wn), resS_by[o], rng,
                                      transmit=T_shard, credit_weight=credit_weight,
                                      agree_min=agree_min, match_tol=match_tol,
                                      match_metric=match_metric,
                                      protect=protectS_by[o], protect_weight=outline_protect_weight)
                        for o in others)
                    for k in viable])
                # maximise primary-wall coverage, minimise secondary-wall damage. Restricting
                # to `viable` (cover>0) is what keeps this safe: a panel that cannot host the
                # shard is EXCLUDED, never rewarded -- so the "aim at nothing scores best"
                # degeneracy that killed the additive placement-scorer penalty cannot recur
                # here, because doing nothing is not on the menu.
                obj = cover[viable] - damage_weight * dmg
                if jpf is not None:                            # B2: pull toward the joint solution's depths
                    jt = np.array([jpf[fam[k][0]][ys, xs].mean() for k in viable])
                    obj = obj + joint_weight * jt
                local_k = int(viable[int(np.argmax(obj))])
            gi, panel = fam[local_k]                            # scatter: which plane hosts this stroke

            # scatter: driven directly by shard-size intent (jitter * spacing). An earlier
            # attempt capped this at a multiple of the panel's own penumbra sigma (the blur
            # radius the renderer already applies before compositing), on the reasoning that
            # jitter under that radius is "free". Empirically that bound is vacuous for a
            # realistic rig: a near-point light (small source_radius) casts a genuinely sharp
            # shadow, so its true penumbra can be 2-3 orders of magnitude smaller than a
            # sensible shard size -- any cap tied to it either collapses the scatter to ~1px
            # or has to be inflated so far it stops meaning anything physically. So intent
            # drives the scatter directly here; the penumbra is instead reported as a
            # diagnostic (how many sigmas of positional error this implies, see
            # `fragments[...]["softening_sigmas"]` and the CLI summary) so the real
            # fidelity/scatter tradeoff is visible per run instead of silently gated on a
            # number that doesn't reflect it.
            jitter_px = max(0, int(round(jitter * spacing)))
            sigma_m = table[(panel.name, family)].penumbra_sigma_m
            sigma_px = sigma_m / max(px_m, 1e-9)
            softening_sigmas = jitter_px / max(sigma_px, 1e-9)

            # DARKNESS/SATURATION LIFT (`intensity_gain`>1, opt-in, default 1.0 = off): the render's
            # intensity-weighted lamination (color.stack_transmit_lut) applies each sheet at only
            # `cvals[ch]` strength, so a deep colour whose channels sit at 0.5-0.8 renders lighter
            # and less saturated than the PHYSICAL full sheet actually casts -- Prussian blue washes
            # to mid-blue. Scaling the recorded per-channel intensities up (clipped at 1.0) deepens
            # those channels toward the full-sheet reality while a genuinely weak channel (e.g. the
            # 0.28 M of a golden-orange, kept low by the hue fix) stays proportionally weakest, so
            # hue is preserved. 1.0 leaves stack_intensity byte-identical to before.
            if intensity_gain != 1.0:
                cvals = {ch: min(1.0, float(v) * intensity_gain) for ch, v in cvals.items()}

            stack_depths.append(len(active))
            shadow_mm2 = float(m.sum()) * px_area_mm2
            for s, ch in enumerate(active):
                mm = m
                dy = int(rng.integers(-jitter_px, jitter_px + 1))
                dx = int(rng.integers(-jitter_px, jitter_px + 1))
                if dy or dx:
                    shifted = np.zeros_like(m)
                    ny = np.clip(ys + dy, 0, Hn - 1)
                    nx = np.clip(xs + dx, 0, Wn - 1)
                    shifted[ny, nx] = True
                    mm = shifted
                panel_wall_stack[gi][s] = np.where(mm, names.index(ch), panel_wall_stack[gi][s])
                # record this channel's CMYK intensity alongside its colour-id so the render
                # can weight the sheet by it (see color.stack_transmit_lut) -- the hue fix.
                panel_wall_intensity[gi][s] = np.where(mm, cvals[ch], panel_wall_intensity[gi][s])
                fragments.append({"wall": family, "panel": panel.name, "channel": ch,
                                  "stack": s, "jitter_px": jitter_px,
                                  "softening_sigmas": softening_sigmas,
                                  "phys_mm2": shadow_mm2 / table[(panel.name, family)].magnification ** 2})

        for gi, panel in fam:
            uv = _panel_pixel_uv(panel, (Hp, Wp))
            ab = H.apply_homography(table[(panel.name, family)].H_pw, uv)
            col = ab[..., 0] / max(wall.width, 1e-9) * Wn - 0.5
            row = ab[..., 1] / max(wall.height, 1e-9) * Hn - 0.5
            for s in range(S):
                res = ndimage.map_coordinates(panel_wall_stack[gi][s], [row.ravel(), col.ravel()],
                                              order=0, mode="constant", cval=0).reshape(Hp, Wp)
                stack_colorid[s, gi] = res.astype(np.int16)
                resI = ndimage.map_coordinates(panel_wall_intensity[gi][s], [row.ravel(), col.ravel()],
                                               order=0, mode="constant", cval=0.0).reshape(Hp, Wp)
                stack_intensity[s, gi] = resI                # nearest, same coords -> stays aligned with colour-id

    opacity = (stack_colorid.max(axis=0) > 0).astype(np.float32)
    resolved = _resolve_collisions(scene, opacity, trim=True)
    for s in range(S):
        stack_colorid[s][opacity == 0] = 0
        stack_intensity[s][opacity == 0] = 0.0
    return stack_colorid, opacity, fragments, resolved, stack_depths, budget_stats, stack_intensity


def panel_stack_pieces(scene, stack_colorid, names, kerf=None):
    """Vectorise EVERY stack slot's colour-id raster -> {panel_name: [(poly, channel, slot)]}.

    Each polygon is a single-channel, single-slot laminated layer; `shard_geom.build_stack_mesh`
    and `export_obj` use the (channel, slot) tag to place it at the right micro depth-offset."""
    from ..raster2vec import features, contours
    from shapely.geometry import Polygon
    kerf = scene.fab.kerf if kerf is None else kerf
    S = stack_colorid.shape[0]
    out = {p.name: [] for p in scene.panels}
    for gi, panel in enumerate(scene.panels):
        Hp, Wp = stack_colorid.shape[-2:]
        pixel = 0.5 * ((panel.u_range[1] - panel.u_range[0]) / Wp
                       + (panel.v_range[1] - panel.v_range[0]) / Hp)
        for s in range(S):
            layer = stack_colorid[s, gi]
            for cid in range(1, len(names)):
                mask = layer == cid
                if not mask.any():
                    continue
                mask = features.enforce_min_feature(mask, scene.fab.min_feature, pixel)
                ps = contours.mask_to_polygons(mask, panel.u_range, panel.v_range)
                if kerf > 0:
                    grown = []
                    for p in ps:
                        b = p.buffer(kerf / 2.0, join_style=2)
                        if b.is_empty:
                            continue
                        grown += [b] if isinstance(b, Polygon) else \
                                 [g for g in getattr(b, "geoms", []) if g.area > 0]
                    ps = grown
                out[panel.name] += [(p, names[cid], s) for p in ps]
    return out


def panel_channel_pieces(scene, colorid, names, kerf=None):
    """Vectorise shards PER CHANNEL from the colour-id map -> {panel_name: [shapely polys]}.

    Each polygon is a single-channel shard (its colour is recovered later by sampling colorid
    at its centroid). Vectorising per channel avoids merging adjacent different-colour shards.
    """
    from ..raster2vec import features, contours
    from shapely.geometry import MultiPolygon, Polygon
    kerf = scene.fab.kerf if kerf is None else kerf
    out = {}
    for gi, panel in enumerate(scene.panels):
        Hp, Wp = colorid[gi].shape
        pixel = 0.5 * ((panel.u_range[1] - panel.u_range[0]) / Wp
                       + (panel.v_range[1] - panel.v_range[0]) / Hp)
        polys = []
        for cid in range(1, len(names)):
            mask = colorid[gi] == cid
            if not mask.any():
                continue
            mask = features.enforce_min_feature(mask, scene.fab.min_feature, pixel)
            ps = contours.mask_to_polygons(mask, panel.u_range, panel.v_range)
            if kerf > 0:
                grown = []
                for p in ps:
                    b = p.buffer(kerf / 2.0, join_style=2)
                    if b.is_empty:
                        continue
                    grown += [b] if isinstance(b, Polygon) else \
                             [g for g in getattr(b, "geoms", []) if g.area > 0]
                ps = grown
            polys += ps
        out[panel.name] = polys
    return out
