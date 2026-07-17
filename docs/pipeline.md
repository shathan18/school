# ShadowArt algorithm pipeline

From **two input images + a scene description** to **shards positioned in 3D space** and
machine-ready cut files. Orchestrated by `cmd_run` in [shadowart/cli.py](../shadowart/cli.py).

```
2 images ──▶ targets ──▶ geometry ──▶ decompose ──▶ render(verify) ──▶ raster2vec ──▶ fabricate
 (A,B)      silhouettes   homographies   shards       predicted        polygons        slots+DXF/SVG
                          + magnif.      per plane     projection                       + 3D shards
```

---

## 1. Load targets
`load_target` ([targets/image_ops.py](../shadowart/targets/image_ops.py))

Each image → grayscale silhouette `[Hn, Wn]` in `[0,1]` at `solve.wall_res`
(**dark = shadow**). Stored as `targets = {"A": …, "B": …}`. A wall value of `1` means
"this spot should be fully dark"; `0` means "leave it lit." Everything downstream is trying
to build these two darkness fields with physical, cuttable material.

## 2. Geometry — projection table
`build_projection_table` ([geometry/projection.py](../shadowart/geometry/projection.py)),
math in [geometry/homography.py](../shadowart/geometry/homography.py)

The physical scene is a room corner: two perpendicular walls, two floor point-lights
(Light A ↔ Wall A, Light B ↔ Wall B), and 8 acrylic panels woven into an egg-crate lattice
— family-A panels are planes at `x = coord`, family-B panels at `y = coord`.

### The homography (why one 3×3 matrix suffices)
A point light `L`, a flat panel, and a flat wall define a **central projection**: the shadow
of a panel point `P` is where the ray `L → P` pierces the wall. Ray–plane intersection is

$$X = L + t\,(P-L), \qquad t = \frac{n\cdot(p_0 - L)}{n\cdot(P-L)}$$

for a wall through `p_0` with normal `n`. Projecting between two planes through a single
centre is a **projective collineation**, so it is captured exactly by a 3×3 homography in
homogeneous coordinates: $[a,b,1]^\top \sim H\,[u,v,1]^\top$. We build it by projecting the
panel's 4 corners to the wall (`project_uv_to_wall`) and solving for `H` with **normalized
DLT + SVD** (`dlt_homography`) — isotropic normalization for conditioning, then the null
vector of the 2n×9 system. Result: one matrix that maps panel-local metres `(u,v)` →
wall-local metres `(a,b)`, plus its inverse `H_wp` for the renderer.

- **Parallel** planes (family-A panel → Wall A) collapse to a pure scaling by the
  magnification.
- **Perpendicular** planes (family-A panel → Wall B) give a strongly **keystoned**
  homography — that skew *is* the cross-talk ghost, handled by the same code path.

### Magnification = the depth/resolution gradient
`m = dist(light, wall) / dist(light, panel)` (the ray parameter `t` at the panel centre).
Panels near the light throw **big, coarse** shadows (a small feature is magnified, so it
blurs); panels near the wall are **~life-size** and carry fine detail. This is the physical
reason shards are "small far, big near."

### Penumbra = the true resolution limit
A real light has finite radius, so shadows aren't sharp. The blur half-width scales as
`source_radius · dist(panel, wall) / dist(light, panel)` and is modelled as a Gaussian σ on
the wall (`_penumbra_sigma_m`, [geometry/psf.py](../shadowart/geometry/psf.py)). A panel
near the light can only render coarse blobs — this caps usable detail per depth plane.

Each panel is mapped to **both** walls → *primary* (builds its image) and *cross-talk*
(oblique ghost), cached with `H_pw`, `H_wp`, `m`, and σ.

## 3. Decompose into shards
`fragment_shards` ([solve/decompose.py](../shadowart/solve/decompose.py)) — default `partition` mode

The goal: split each wall's silhouette so its material is **spread across the family's 4
depth planes** such that (a) each individual panel is unreadable dust, but (b) the *union*
of all planes still tiles the silhouette (so the summed shadow reproduces the image).

Per family (A, then B):
1. **Fragment** the silhouette into organic shards. Jittered **blue-noise seeds** inside the
   mask → nearest-seed **Voronoi cells**. Cells over `fragment_max_area` are recursively
   re-seeded and cut (`_split_large`); cells under `fragment_min_area` are dropped, then
   every orphaned pixel is grown into its **nearest** kept shard via a distance transform —
   so coverage stays hole-free (`_fragments`).
2. **Scatter** each shard onto **one** of that family's depth planes. Assignment is a
   weighted random draw; weights `w ∝ 1 + depth_bias·(m_max − m)/range` optionally bias
   count toward the **near** (low-magnification) panels. Spreading across planes is what
   makes any single panel meaningless.
3. **Erode** by `shard_gap` so pieces read as separate floating fragments.
4. **Back-project** each panel's accumulated wall mask through `H_pw` onto the panel grid
   (nearest-neighbour `map_coordinates`) → hard `opacity[P, Hp, Wp]` in {0,1}.
5. **Resolve weave collisions** (`_resolve_collisions`): family-A panel `x=xA` and family-B
   panel `y=yB` physically cross along a vertical line; where **both** hold material there,
   a thin clearance band (width from `material_thickness + joint_clearance`) is trimmed from
   family B so the parts can slide together.

*(Alternative `optimize` mode — see §7 — does inverse rendering instead of partitioning.)*

## 4. Forward render — verify
`Renderer` ([forward/renderer.py](../shadowart/forward/renderer.py)) — differentiable (PyTorch), the single source of truth

This one model is simultaneously the **physics simulator**, the **optimisation objective**,
and the **preview**. For each wall (lit by its own light):

1. **Warp** every panel's opacity onto the wall. Precompute a sampling grid by pushing each
   wall pixel back through `H_wp` to panel coords, then `F.grid_sample` (bilinear).
2. **Blur** each warped contribution by that panel's penumbra Gaussian (separable 1-D
   convs) — near-light panels blur more.
3. **Composite by transmittance.** Light passing straight through independent occluders
   multiplies: $T = \prod_p (1-\alpha_p)$, so wall **darkness** $D = 1 - \prod_p (1-\alpha_p)$.
   Opacities compose multiplicatively, *not* additively — two half-opaque shards on one ray
   give $1-0.5^2 = 0.75$, not `1.0`.

Because **every** panel is warped onto **both** walls, cross-talk
([forward/crosstalk.py](../shadowart/forward/crosstalk.py)) falls out automatically — a
plain two-wall loss already fights it. Outputs: predicted darkness for the preview, per-wall
MSE, and shards-only **IoU** (reconstruction fidelity excluding cross-talk).

Colour mode swaps the composite for a **subtractive** product of per-channel
transmittances (`render_color_np`), so overlapping C+M shards read blue, etc.

## 5. Raster → vector
`raster_to_pieces` (cli) → [raster2vec/](../shadowart/raster2vec/)

Turn each panel's opacity field into cuttable outlines:
1. **Threshold** the opacity to a binary mask.
2. **Enforce minimum feature** (`enforce_min_feature`): morphological **open** (delete solid
   islands smaller than the smallest cuttable detail) then **close** (delete gaps/holes that
   small), using a disk of radius `min_feature/2`. This is where the depth→resolution limit
   becomes physical — you can't cut what the laser/penumbra can't hold.
3. **Contour** the mask → shapely polygons in panel metres.
4. **Kerf offset** (`kerf_offset`, pyclipper): grow each ring by half the cut width so the
   finished part ends up nominal size after the laser eats the edge.

## 6. Fabricate + place in space
`build_panel_drawings`, `nesting.nest`, `export_dxf/svg`, `build_interactive`

- **Weave slots** ([fabricate/joints.py](../shadowart/fabricate/joints.py)): at every A/B
  crossing, cut a **half-lap** slot (`thickness/2` deep) into each panel so the egg-crate
  slides together — rigid flat panels can't literally interleave.
- **Nest** panels onto stock sheets with greedy shelf packing; oversize panels are flagged.
- **Export** `OUTLINE` / `SLOT` / `PIECE` layers to DXF + SVG per panel.
- **Shard position in space is set by its panel:** family-A shard on plane `x = coord`
  (local `u,v → y,z`); family-B shard on `y = coord` (local `u,v → x,z`). The panel's depth
  fixes *where* the shard floats; its 2-D shape is the opaque region from §3.

---

## 7. Alternative: `optimize` mode (inverse rendering)
[solve/optimizer.py](../shadowart/solve/optimizer.py) + [solve/losses.py](../shadowart/solve/losses.py)

Instead of partitioning, treat the opacity fields as free variables and **fit** them to both
targets with gradient descent (Adam). Optimise **logits** `θ` with `α = sigmoid(θ)` (keeps
opacity in `[0,1]`), minimising

$$\mathcal{L} = \underbrace{\|D_A - t_A\|^2 + \|D_B - t_B\|^2}_{\text{reconstruction}}
  + \lambda_{\text{sparse}}\,\overline{\alpha}
  + \lambda_{\text{TV}}\,\mathrm{TV}(\alpha)
  + \lambda_{\text{ct}}\,E_{\text{crosstalk}}.$$

- **reconstruction** — MSE against both walls (the renderer *is* the objective).
- **sparsity** — mean opacity → fewer/floating pieces, less material.
- **total variation** — smoother regions, fewer slivers before halftone.
- **cross-talk** — optional explicit penalty on ghost darkness (usually implicit already).

Sharper than `partition`, but each panel becomes a faded copy of the whole picture (reads as
flat planes), which is why `partition` is the default artistic mode. Always **re-score the
binarised result** — the continuous optimum flatters you.

---

## Why two images fit on one set of pieces
- **90° light/wall geometry** → each light's rays hit its own wall face-on, so family-A
  pieces paint Wall A and family-B pieces paint Wall B through **distinct homographies**.
- **Image shattered across depth planes** → each panel is meaningless dust; the picture only
  exists in the *summed, multiplicative* shadow. This is the encryption: one plane reveals
  nothing.
- **Renderer + cross-talk term** simulate (and correct) both projections — including the
  keystoned ghost each family throws on the *other* wall — before anything is cut.
- **Magnification + penumbra** dictate the shard-size budget per depth: coarse near the
  light, fine near the wall.

## Greedy heuristics used
See [greedy-partitioning.md](greedy-partitioning.md): shard re-split/merge, CMYK layer
budgeting, and shelf nesting.
