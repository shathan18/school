# Greedy algorithms in shard partitioning & layout

The pipeline uses greedy heuristics in three places: **splitting** the silhouette into
shards, **budgeting** how many colour layers each shard gets, and **packing** the finished
panels onto stock sheets. Each makes a locally-best choice and never backtracks — fast,
simple, and good enough for fabrication.

---

## 1. Fragmenting the silhouette into shards

`solve/decompose.py` → `_fragments` / `_split_large`

Blue-noise seeds define Voronoi cells over the silhouette. Two greedy passes then fix cells
that are the wrong size:

- **Too big → greedily re-split.** A cell over `max_px` is recursively re-seeded and cut
  until every piece fits (locally: "this cell is too big, split it now").
- **Too small → greedily absorb.** Sub-`min_px` slivers are dropped, then every orphaned
  pixel is grown into its *nearest* kept shard so coverage stays complete (no holes).

```python
# recursively split a too-big component into sub-cells <= ~max_px
def _split_large(comp, max_px, rng, depth=0):
    if int(comp.sum()) <= max_px or depth >= 3:      # small enough -> stop
        return [comp]
    seeds = _seed_points(comp, np.sqrt(max_px) * 0.85, rng)
    lab = _voronoi_labels(comp, seeds)
    out = []
    for k in range(len(seeds)):                      # greedily cut, recurse on each piece
        piece, n = ndimage.label(lab == k)
        for c in range(1, n + 1):
            out += _split_large(piece == c, max_px, rng, depth + 1)
    return out

# grow kept fragments into leftover slivers -> nearest wins (greedy, complete coverage)
inds = ndimage.distance_transform_edt(frag_id == 0, return_indices=True,
                                      return_distances=False)
frag_id = np.where(holes, frag_id[tuple(inds)], frag_id)
```

**Greedy choice:** each cell is fixed on its own (split if big, merge into nearest if tiny)
without re-optimising the whole tiling.

---

## 2. Budgeting colour layers per shard

`solve/decompose.py` → `_layer_counts`

Each shard's dominant colour maps to C/M/Y/K intensities, which become integer layer counts.
The total must fit the per-shard plane **budget** (number of depth planes). When it overflows,
we greedily remove one layer from the **densest** channel and repeat.

```python
def _layer_counts(cvals, max_layers, budget):
    counts = {ch: min(max_layers, round(cvals[ch] * max_layers)) for ch in CMYK}
    total = sum(counts.values())
    while total > budget:                    # over budget?
        ch = max(counts, key=lambda k: counts[k])   # take from the densest channel
        counts[ch] -= 1
        total -= 1
    return counts
```

**Greedy choice:** always shave the channel that currently has the most layers — cheapest
single fix toward the budget, no lookahead.

---

## 3. Packing panels onto stock sheets

`fabricate/nesting.py` → `nest`

Classic **greedy shelf packing**: place panels left-to-right in a row; when the next panel
would overflow the sheet width, start a new shelf; when it overflows the height, start a new
sheet. Panels never move once placed.

```python
for d in drawings:
    w, h = d.u_range[1] - d.u_range[0], d.v_range[1] - d.v_range[0]
    if cx + w > sw - margin:          # doesn't fit this shelf -> wrap down
        cx = margin; cy += shelf_h + margin; shelf_h = 0.0
    if cy + h > sh - margin:          # doesn't fit this sheet -> new sheet
        sheet += 1; cx = margin; cy = margin; shelf_h = 0.0
    placements.append(Placement(d.name, sheet, cx - d.u_range[0], cy - d.v_range[0], ...))
    cx += w + margin                  # advance the cursor
    shelf_h = max(shelf_h, h)         # shelf height = tallest panel so far
```

**Greedy choice:** put each panel in the first spot the cursor reaches; wrap only when it
won't fit. Not optimal bin-packing, but predictable and O(n).

---

### Why greedy here

All three problems (optimal tiling, optimal layer assignment, optimal 2D bin-packing) are
expensive to solve exactly and don't need to be: shards just have to be a sensible size,
layers must respect the plane budget, and panels must fit the sheet. A local-best-choice
heuristic delivers that in one linear pass.
