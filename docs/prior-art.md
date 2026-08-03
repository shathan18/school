# Prior art, and what this project actually took from it

The brief asked for the literature to be mined rather than for the problem to be re-derived.
This is the audit: what each line of work contributes, and — more usefully — which ideas were
tried here and **failed**, because a reading list that only records the hits is not evidence
of having read anything.

Each entry ends with a **verdict** tag:

- **ADOPTED** — in the shipped build, with the measured gain.
- **REFUTED** — tried here, measured, did not help. Kept so it is not re-tried.
- **NOT APPLICABLE** — good work, wrong regime for this piece, with the reason.
- **OPEN** — plausible, not yet tested.

---

## 1. Shadow art as inverse design

### Mitra & Pauly, _Shadow Art_ (SIGGRAPH Asia 2009)

The founding paper: given several target silhouettes and light directions, carve a single
volume whose shadows match. Their central move is to start from the **visual hull** — the
intersection of the back-projected target cones — because that is the largest object whose
shadows are all subsets of the targets, and then to deform toward it under a smoothness prior.
They also make the point this project keeps running into: for many target triples **no exact
solution exists**, and the honest response is to relax the targets slightly rather than to
report a fit that is not there.

_What this build shares_: nothing at the representation level — a visual hull is a solid, this
is a set of flat sheets. _What it shares at the objective level_: the admission that the three
views fight each other, which is precisely what the cross-talk term prices.

**Verdict: NOT APPLICABLE (representation).** A visual-hull carve assumes a connected solid
occupying the whole intersection volume. The 30 cm turntable is 18 thin sheets on a triangular
grid; there is no volume to carve, and the manufacturing constraint (laser-cut flat stock) is
what forces that. The _relaxation_ insight is adopted implicitly: the targets here are
grayscale, so partial satisfaction is representable rather than being a failure.

### Baran et al., _Manufacturing Layered Attenuators for Multiple Prescribed Shadow Images_

(Eurographics 2012)

The closest relative, and the one that matters most. Stacked **semi-transparent** layers,
attenuation multiplying through the stack, solved for prescribed grayscale (not binary)
shadows. Two ideas carry directly:

1. **Multiplicative compositing is the physics**, and it is what makes grayscale shadow art
   tractable at all: `T_total = ∏ T_layer`. This repo's renderer already composites in
   transmittance (`render_color_np` returns transmittance in [0,1]), so `T_total = T_own ×
T_stray` holds exactly — which is the identity the whole cross-talk analysis rests on.
2. **Work in optical density, not transmittance.** Because transmittances multiply, densities
   `D = −log T` _add_, and a stack is linear in density. Layer levels should therefore be
   spaced evenly in `D`, not in `T`.

Point 2 was tested here directly and is one of the clearest results in the project:

| engrave levels (transmittance) | spacing               | mean IoU  |
| ------------------------------ | --------------------- | --------- |
| 0.60 / 0.30 / 0.10             | ~even in density      | **0.819** |
| 0.56 / 0.32 / 0.18             | ~even in density      | 0.817     |
| 0.75 / 0.50 / 0.25             | even in transmittance | 0.785     |

**Verdict: ADOPTED.** Density-even level allocation is worth **+0.034 mean IoU** over the
transmittance-even choice that looks more natural on a spec sheet. The shipped levels are
0.60 / 0.30 / 0.10.

### ShadowPix (Bermano et al., 2012) and self-shadowing relief

Height-field reliefs that produce several images under different grazing light directions,
using _self-shadowing_ of surface microstructure rather than transmission.

**Verdict: NOT APPLICABLE.** It needs grazing light and an opaque relief. This piece is
transmissive, lit by a near-point source at a normal-ish incidence, and must read from a
distance of metres. The one transferable idea — that neighbouring surface elements can be made
to interfere constructively for one view and destructively for another — is already what the
damage-aware host selection does in a different geometry.

---

## 2. Attenuation-layer displays

### Wetzstein et al., _Layered 3D_ / tomographic and NTF-based attenuator stacks (2011–2012)

Multi-layer attenuators driven by **non-negative tensor factorisation**: the light field is a
tensor, the layers are its rank-limited factors, and the solve is a multiplicative-update NTF.

Two directly relevant lessons:

1. **More layers, closer together, beat fewer layers spread out** — the rank of what a stack
   can represent grows with layer count, and tight spacing keeps the layers close to
   co-registered so they compose rather than blur past each other.
2. Content **beyond the depth of field of the stack** cannot be represented and should be
   pre-filtered out rather than fought.

Lesson 1 is the single largest measured gain in this project:

| sheets per family | pitch     | mean IoU  | cross-talk cost |
| ----------------- | --------- | --------- | --------------- |
| 2                 | 60 mm     | 0.746     | +0.076          |
| 3                 | 30 mm     | 0.796     | +0.014          |
| 4                 | 20 mm     | 0.810     | −0.014          |
| **6**             | **20 mm** | **0.816** | **−0.033**      |
| 10                | 25 mm     | 0.821     | +0.020          |

**Verdict: ADOPTED — +0.070 mean IoU**, the biggest single lever found. Note the sign change
in the last column: past ~4 sheets per family the cross-talk cost goes **negative**, i.e. the
all-light render beats the shards-only render. The stray light stopped being noise and became
part of the picture, which is exactly the NTF picture of a stack where layers co-operate. The
knee is at 18 sheets; 30 sheets buys a further 0.005 and is not worth the material.

**Verdict on lesson 2: REFUTED — measured, and wrong in every cell.** The reasoning above was
that penumbra σ is 3.6–4.6 mm on the wall while the min feature projects to 32 mm, so there
must be target energy above the reproducible band that the solver is wasting shards on.
Pre-filtering the targets with a Gaussian before the solver sees them (`BuildConfig.target_blur_px`,
sweep `bandlimit`; scoring always against the _unfiltered_ targets) gives a monotone loss on
every axis:

| target blur (px) | 0         | 1     | 2     | 3     | 4     | 6     |
| ---------------- | --------- | ----- | ----- | ----- | ----- | ----- |
| mean IoU         | **0.819** | 0.811 | 0.800 | 0.784 | 0.776 | 0.756 |
| edge fidelity    | **0.556** | 0.448 | 0.279 | 0.163 | 0.124 | 0.130 |

There is no interior optimum, not even at 1 px. The premise was wrong in two places: the
32 mm figure was the _cut-out_ min feature, which as §5 records never applied to engraved
tone regions at all; and a shard boundary is a step edge whose reproducible content is set by
the penumbra (≈1.3 panel px), not by the shard size. Asking the solver to fit a blurred target
does not stop it spending shards on fine structure — it just aims that structure at the wrong
place. **Do not re-run this.**

---

## 3. Halftoning and dithering

### Floyd–Steinberg error diffusion; Ulichney's void-and-cluster; blue-noise masks

The relevant result is not any particular algorithm but the **spectral** one: for a fixed
number of quantisation levels, the visibility of quantisation error depends far more on how
the error is _distributed in frequency_ than on how large it is. Blue-noise error — energy
pushed to high frequencies — is nearly invisible; low-frequency (clustered) error reads as
blotches and banding.

This piece quantises to four tones, so it is a halftoning problem whether or not it is called
one. And the failure mode currently visible in the renders — **vertical dark streaking**, which
is low-frequency structured error — is exactly the failure blue-noise theory predicts when the
error is allowed to correlate along one axis. The streaks run parallel to the sheet stacking
direction, which is precisely the axis along which the shard partition is free to correlate.

**Verdict: OPEN, and the highest-value remaining work.** The lever is not "add dithering" but
"add a term that penalises _low-frequency_ residual specifically", so the solver is pushed to
spend its error budget at frequencies the eye discounts. A plain spatial-smoothness penalty is
the wrong tool here and is known to be: this repo previously measured a heavy de-streak pass
dropping IoU from 0.83 to 0.55.

---

## 4. Radiometric compensation (projector–camera systems)

Standard practice for projecting onto non-white surfaces: measure what the surface contributes,
then divide it out of the sent image so the product lands on the intended result.

Applied here, the analogue is exact. Each view sees `T_own × T_stray`; the decomposer fits
`T_own` to the target while being told nothing about `T_stray`, which is why reconstructions
come out systematically too dark (measured foreground area 0.30 against a target of 0.21). So:
solve, measure the stray transmittance, re-aim at `target / T_stray`, re-solve.

Implemented in `pearl3_baseline.precompensate_targets`. Measured fixed-point iteration:

```
0.690  ->  0.598  ->  0.665  ->  0.641   (mean IoU)
```

**Verdict: REFUTED.** It oscillates, and the reason is specific and instructive: in a
projector–camera system the "surface" is passive and independent of what you send. Here it is
not. 94% of shards deliberately serve two or more views, so `T_stray` is _made of the same
shards as `T_own`_ — dividing it out double-counts the sharing the design is built on, every
view lightens simultaneously, and the map has no reason to contract. Kept behind a damping
gain and a best-pass selector, but not shipped. Superseded entirely by the tight-stack lever,
which fixes the same darkness problem by making the stray light constructive instead of by
subtracting it.

---

## 5. Computational fabrication: the raster-to-object gap

The consistent lesson across the fabrication literature is that the score of the simulation is
not the score of the object, and the gap must be _measured_, not assumed small. This repo has
its own scar: re-rendering from vectorised polygons by scanline-refilling marching-squares
contours scored IoU 0.9154, while testing pixel **centres** with a proper point-in-polygon test
scored 0.9776 on the identical geometry. Six points of IoU were an artefact of the verification
code.

**Verdict: ADOPTED as process.** `pearl3_fab.py` gate B re-rasterises the exported polygons via
`shapely.contains_xy` on pixel centres and re-renders, and additionally collapses each region's
engrave intensity to a single value — because a raster engrave pass fires at one power over a
region, and the solver's freedom to shade a shard's interior pixel-by-pixel does not survive
contact with the machine.

That gate then paid for itself twice, and both times by catching a **constraint that had been
inherited from the wrong manufacturing process**:

**5a. The minimum feature size was the wrong constraint.** A stage-2 tone re-quantisation
(§7) improved every raster metric — mean IoU 0.8206 → 0.8283, SSIM 0.7528 → 0.8026, RMSE
0.1399 → 0.0963 — and then scored **0.4154** through gate B. The export was deleting the gain,
because `enforce_min_feature` was applying a 5 mm floor to the engraved tone regions. But 5 mm
is the limit for a shard **cut out** of a sheet, which has to survive as a self-supporting
physical object. An engraved region is a mark on a sheet that stays whole; its only limit is
the beam spot, ~0.2 mm, which is _under half a panel pixel_. The structural limit still governs
the `CUT_OUTLINE` and `CUT_SLOT` layers, where it belongs. Separating the two
(`BuildConfig.engrave_min_feature`) is what made the stage-2 gain real.

**5b. Marching squares is half a pixel small.** `contours.mask_to_polygons` extracts the 0.5
level set, which runs through the **centres** of the boundary pixels — so every exported region
is inset by half a pixel on each side. The resulting area error is proportional to total
perimeter, which is why it hid so well: on stage 1's 281 large regions it cost +0.0037 mean
IoU, and on stage 2's 5402 small regions the _identical exporter at identical settings_ cost
+0.0496. Compensating with a half-pixel outward grow brought that to **+0.0011**. This is a
registration correction and not kerf compensation — engraving removes no material at the
boundary and there is no kerf to offset.

The general lesson is worth more than either fix: **a constraint carried over from a different
process is invisible, because it never throws an error — it just quietly caps quality.** And a
verification gate is only useful if something can fail it; both of these were found by a gate
that every raster metric said should have passed.

---

## 6. Derivative-free optimisation

The brief asked for several methods to be compared rather than for one to be asserted. The
relevant background is the standard one — CMA-ES (Hansen) for its covariance adaptation on
coupled variables, TPE (Bergstra et al.) for expensive black boxes, simulated annealing
(Kirkpatrick) for multi-modal landscapes, GAs for recombining partial solutions — plus the
uncomfortable and well-replicated finding that **random search is a strong baseline** in
moderate dimension when only a few axes matter (Bergstra & Bengio 2012).

That last point is why random search is included here as a control arm rather than as a
courtesy. See `docs/30x30-optimization.md` for the equal-budget bake-off results.

---

## Summary of transfer

| Source idea                                         | Verdict           | Measured                             |
| --------------------------------------------------- | ----------------- | ------------------------------------ |
| Layered attenuators: work in optical density        | ADOPTED           | +0.034 mean IoU                      |
| Attenuator stacks: many layers, tight pitch         | ADOPTED           | +0.070 mean IoU                      |
| Cross-talk-aware assignment (damage pricing)        | ADOPTED           | +0.058 mean IoU                      |
| Pixel-centre point-in-polygon verification          | ADOPTED (process) | avoids a −0.06 artefact              |
| Engrave limit ≠ cut-out limit (beam spot, not 5 mm) | ADOPTED           | unlocked +0.021 shipped IoU          |
| Half-pixel contour registration                     | ADOPTED           | gate B cost 0.0496 → 0.0011          |
| Two-stage solve: geometry, then tone                | ADOPTED           | +0.021 IoU, +0.039 SSIM, −0.033 RMSE |
| Radiometric pre-compensation                        | REFUTED           | oscillates, −0.05                    |
| Visual-hull carving                                 | NOT APPLICABLE    | no solid to carve                    |
| Self-shadowing relief (ShadowPix)                   | NOT APPLICABLE    | wrong optics                         |
| Blue-noise / frequency-shaped error                 | OPEN              | targets the visible streaking        |
| Band-limiting targets to the optical passband       | REFUTED           | monotone loss, no interior optimum   |
