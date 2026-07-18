"""
Read a painting as OBJECTS, not as one silhouette.

The solver can already fragment each labelled region on its own Voronoi (`region_masks` ->
`decompose._fragments_regions`), so no shard crosses a region boundary and the outlines stay
crisp. What was missing is a GENERAL way to produce those labels: until now they were hand-coded
per image (lemons vs dish, sunflower heads).

`segment_objects` does it automatically:
  1. posterise the subject to `k` colours (k-means on a subsample -- flat/posterised art and
     Van Gogh's blocked colour both cluster well),
  2. connected components of each colour  ->  candidate objects
     (each sunflower head, the vase, the beard, the coat, a star, the river band ...),
  3. absorb specks below `min_frac` into their nearest neighbour so coverage stays hole-free,
  4. optionally watershed-split blobs that are really several touching objects of one colour
     (adjacent flowers, a cluster of stars) via distance-transform peaks + nearest-seed.

`face_like_regions` marks the regions a portrait's face occupies (skin-ish colour inside an
upper-centre ROI) so they can be given a finer shard spacing via `region_scales`.

Import from an experiment script; not part of the shipped pipeline.
"""
import numpy as np
from scipy import ndimage

RNG = np.random.default_rng(0)


def _kmeans(px, k, iters=12):
    if len(px) < k:
        return px.copy()
    cen = px[RNG.choice(len(px), k, replace=False)]
    for _ in range(iters):
        lab = ((px[:, None, :] - cen[None, :, :]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                cen[j] = px[lab == j].mean(0)
    return cen


def posterise_labels(rgb, mask, k=10, sample=20000):
    """Colour-cluster id per subject pixel (0 outside the subject)."""
    px = rgb[mask].reshape(-1, 3)
    if len(px) == 0:
        return np.zeros(rgb.shape[:2], int)
    sub = px[RNG.choice(len(px), min(sample, len(px)), replace=False)]
    cen = _kmeans(sub.astype(float), k)
    ids = ((px[:, None, :] - cen[None, :, :]) ** 2).sum(-1).argmin(1) + 1
    out = np.zeros(rgb.shape[:2], int)
    out[mask] = ids
    return out


def split_touching(mask, peak_frac=0.09, min_dist_frac=0.025):
    """Separate touching blobs of one colour (adjacent flower heads, a clump of stars):
    distance-transform peaks become seeds, every pixel takes its nearest seed."""
    H, W = mask.shape
    dist = ndimage.distance_transform_edt(mask)
    win = max(3, int(peak_frac * min(H, W)) | 1)
    peaks = mask & (dist == ndimage.maximum_filter(dist, size=win)) & \
        (dist > min_dist_frac * min(H, W))
    pl, n = ndimage.label(peaks)
    if n <= 1:
        return mask.astype(int)
    seed = np.zeros((H, W), int)
    for i, (cy, cx) in enumerate(ndimage.center_of_mass(peaks, pl, range(1, n + 1)), start=1):
        seed[int(round(cy)), int(round(cx))] = i
    inds = ndimage.distance_transform_edt(seed == 0, return_indices=True, return_distances=False)
    return np.where(mask, seed[tuple(inds)], 0)


def segment_objects(rgb, mask, k=10, min_frac=0.004, split=True, max_objects=60):
    """Label map (0 = background, 1..N = objects) of the subject's perceptual objects."""
    H, W = rgb.shape[:2]
    pl = posterise_labels(rgb, mask, k=k)
    min_px = max(40, int(min_frac * max(int(mask.sum()), 1)))
    out = np.zeros((H, W), int)
    nxt = 0
    for c in range(1, pl.max() + 1):
        comp, n = ndimage.label(pl == c)
        for i in range(1, n + 1):
            blob = comp == i
            if blob.sum() < min_px:
                continue                                   # speck: absorbed in the grow-back below
            parts = split_touching(blob) if split else blob.astype(int)
            for p in range(1, int(parts.max()) + 1) if split else [1]:
                piece = (parts == p) if split else blob
                if piece.sum() < min_px:
                    continue
                nxt += 1
                out[piece] = nxt
    if nxt == 0:
        out[mask] = 1
        return out
    # grow the kept objects over dropped specks so the subject stays fully covered
    holes = mask & (out == 0)
    if holes.any():
        inds = ndimage.distance_transform_edt(out == 0, return_indices=True, return_distances=False)
        out = np.where(holes, out[tuple(inds)], out)
    out[~mask] = 0
    if nxt > max_objects:                                  # keep the biggest, merge the rest in
        counts = np.bincount(out.ravel()); counts[0] = 0
        keep = set(np.argsort(counts)[::-1][:max_objects].tolist())
        drop = mask & ~np.isin(out, list(keep))
        out[drop] = 0
        inds = ndimage.distance_transform_edt(out == 0, return_indices=True, return_distances=False)
        out = np.where(mask & (out == 0), out[tuple(inds)], out)
        out[~mask] = 0
    return out


def _edge_strength(rgb, blur=1.2):
    """Perceptual edge map: gradient magnitude over all three channels, lightly blurred so
    brushwork texture doesn't read as a boundary."""
    sm = np.stack([ndimage.gaussian_filter(rgb[..., c], blur) for c in range(3)], -1)
    g = np.zeros(rgb.shape[:2])
    for c in range(3):
        g += np.hypot(ndimage.sobel(sm[..., c], 0), ndimage.sobel(sm[..., c], 1)) ** 2
    g = np.sqrt(g)
    p = np.percentile(g, 97) or 1.0
    return np.clip(g / p, 0, 1)


def _adjacency(lab, edge):
    """{(a,b): (mean edge strength on their shared border, border length)} for touching regions."""
    acc = {}
    for A, B, E in ((lab[:, :-1], lab[:, 1:], np.maximum(edge[:, :-1], edge[:, 1:])),
                    (lab[:-1, :], lab[1:, :], np.maximum(edge[:-1, :], edge[1:, :]))):
        m = (A != B) & (A > 0) & (B > 0)
        if not m.any():
            continue
        a, b, e = A[m], B[m], E[m]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo.astype(np.int64) * 1000000 + hi.astype(np.int64)
        uniq, inv = np.unique(key, return_inverse=True)
        s = np.zeros(len(uniq)); np.add.at(s, inv, e)
        n = np.zeros(len(uniq)); np.add.at(n, inv, 1.0)
        for k, si, ni in zip(uniq, s, n):
            pair = (int(k // 1000000), int(k % 1000000))
            ps, pn = acc.get(pair, (0.0, 0.0))
            acc[pair] = (ps + si, pn + ni)
    return {k: (s / n, n) for k, (s, n) in acc.items() if n > 0}


def segment_objects_edges(rgb, mask, k=26, target=22, min_frac=0.004,
                          max_region_frac=0.45, split=True):
    """Object map built on EDGES, not colour alone.

    Colour clustering alone gets this wrong in both directions: a smoothly shaded vase breaks
    into several colour patches, while two touching flowers of the same yellow merge into one.
    So instead:
      1. oversegment (fine colour clusters -> connected components, touching blobs split),
      2. measure the real EDGE STRENGTH along every shared border,
      3. greedily merge the WEAKEST borders first -- shading inside one object has no edge and
         merges away, while a genuine object boundary survives -- until `target` objects remain.
    `max_region_frac` stops everything collapsing into one blob.
    """
    H, W = rgb.shape[:2]
    # OVERsegment first: the floor here must be tiny, otherwise the small pieces are absorbed
    # before the merge stage can run and we end up with a handful of blobs regardless of
    # `target`. The real size control is the merging below, not this floor.
    lab = segment_objects(rgb, mask, k=k, min_frac=min_frac * 0.05, split=split, max_objects=600)
    n0 = int(lab.max())
    if n0 <= target:
        return lab
    edge = _edge_strength(rgb)
    sizes = np.bincount(lab.ravel(), minlength=n0 + 1).astype(float)
    parent = list(range(n0 + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    adj = _adjacency(lab, edge)
    order = sorted(adj.items(), key=lambda kv: kv[1][0])       # weakest border first
    cap = max_region_frac * max(float(mask.sum()), 1.0)
    count = n0
    for (a, b), (w, _n) in order:
        if count <= target:
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if sizes[ra] + sizes[rb] > cap:                        # don't swallow the whole image
            continue
        parent[rb] = ra
        sizes[ra] += sizes[rb]
        count -= 1
    out = np.zeros_like(lab)
    remap, nxt = {}, 0
    for v in range(1, n0 + 1):
        r = find(v)
        if r not in remap:
            nxt += 1; remap[r] = nxt
        out[lab == v] = remap[r]
    return out


def face_like_regions(rgb, labels, roi_rows=(0.50, 0.98), roi_cols=(0.18, 0.85),
                      min_overlap=0.35):
    """Which object labels are the portrait's FACE: skin-ish (warm, mid-bright, not too
    saturated) AND mostly inside an upper-centre ROI. Targets are stored flipped (image top =
    HIGH row index), so the head sits at high rows. Heuristic, not face detection."""
    H, W = rgb.shape[:2]
    roi = np.zeros((H, W), bool)
    roi[int(roi_rows[0] * H):int(roi_rows[1] * H), int(roi_cols[0] * W):int(roi_cols[1] * W)] = True
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    warm = (R > 0.45) & (R >= G - 0.02) & (G >= B - 0.05)
    out = []
    for v in range(1, int(labels.max()) + 1):
        m = labels == v
        if not m.any():
            continue
        if (m & roi).sum() / m.sum() >= min_overlap and warm[m].mean() > 0.5:
            out.append(v)
    return out


def overlay(rgb, labels, alpha=0.55, seed=3):
    """Random-colour overlay of the object map, for previews."""
    rng = np.random.default_rng(seed)
    out = np.clip(rgb, 0, 1).copy()
    for v in range(1, int(labels.max()) + 1):
        m = labels == v
        if m.any():
            out[m] = (1 - alpha) * out[m] + alpha * rng.uniform(0.15, 1.0, 3)
    return out
