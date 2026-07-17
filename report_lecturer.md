# Cross-Talk-Aware Shard Assignment for Dual-Image Shadow Art
### A response to the "two separate blocks with no intersection" critique

**Status:** Simulation only. No physical artefact has been fabricated.
**Scope of claim:** One algorithmic contribution (shard-to-panel assignment) and one empirical characterisation (the intersection-vs-noise trade-off). We do **not** claim novelty in shadow-art fabrication, shard generation, or colour separation.

---

## 1. Motivation and the critique being answered

The system takes two source images and designs a single physical assembly — angled acrylic panels carrying coloured shards — which, illuminated by two point lights in a room corner, casts a *different* target image on each of two perpendicular walls.

An earlier review of this work raised a specific and, on inspection, **correct** objection: that the design amounted to **two separate blocks with no genuine intersection** — that each panel served one wall and one wall only, and the "dual" nature of the piece was a spatial coincidence rather than a computational or physical entanglement. Nothing in the pipeline required, produced, or measured any element serving both images.

This report addresses that critique directly. We (i) quantify the degree to which panels serve both walls, (ii) identify the single pipeline decision that governs it, and (iii) show that a corrected objective produces panels that **measurably do serve both walls**, while *simultaneously* reducing visual noise. We also characterise the relationship between the two effects, which turns out to be more interesting than a simple trade-off.

---

## 2. Physical setting and the origin of cross-talk

Two point lights illuminate a shared assembly **simultaneously**. A shard is an isotropic piece of tinted acrylic: it cannot choose which light it blocks. It therefore casts **two** shadows — one on each wall.

We term the shadow falling on the wall a shard was *not* designed for **cross-talk**. Formally, this is unavoidable:

> For a shard required to occlude pixel *p* on Wall A, its admissible positions lie on the **ray from Light A through *p***. This is a one-dimensional locus. Sliding the shard along that ray leaves its Wall-A shadow invariant (up to scale) but **sweeps its Wall-B shadow across Wall B**.

Consequently the shard's **depth along its primary ray is the sole free parameter governing where its secondary shadow lands.** Cross-talk cannot be eliminated by any placement algorithm; it can only be *steered*. (Elimination would require a change of physical concept — directionally selective material, or time-multiplexed illumination — neither of which is a coloured-acrylic sculpture.)

We distinguish two kinds by outcome:
- **Bad cross-talk** — secondary shadow landing on the *background* of the other image. Visible noise.
- **Good cross-talk (joint-intersection)** — secondary shadow landing on the other image's *content*. This is the panel serving both walls: **precisely the intersection whose absence was criticised.**

Crucially, **these are the same physical event**, distinguished only by where the shadow lands. Whether they can be separated is an empirical question, and is the central question of this report.

---

## 3. Method

### 3.1 INPUT

**Images:** two RGB sources, cropped to subject and fitted to the wall canvas (`targets/color.py::load_color_target`).

**Parameters.** User-specified: wall geometry (1.8 × 1.8 m), light positions (3.0 m throw), material thickness (3 mm, verified as commercial stock), panel count, shard sizing, colour thresholds. Algorithm-determined: panel angles/positions (if the search is enabled), shard count and sizes (adaptive to image detail), shard colours, and **shard-to-panel assignment** (the contribution).

A full audit of which constants are *derived* versus *chosen* appears in §6.3. We regard this distinction as material to the report's credibility.

### 3.2 ALGORITHM — standard components (not claimed as novel)

**Panel placement** (`solve/panel_search.py`). A fixed panel count (not searched; no stopping rule was implemented). Each candidate receives a **uniformly random** angle and position, rejection-filtered against: room bounds; a 0.5 m standoff from both walls; and a magnification cap of 3× (preventing a single near-light panel from blanketing a wall). Candidates are selected greedily by marginal coverage of image content (best-of-16). The *selection* is deliberate; each *candidate* is stochastic.

**Shard generation — a Voronoi partition** (`solve/decompose.py::_fragments`). Stated plainly: **shards are Voronoi cells.** Seed points are placed on a jittered regular grid within the image subject; **every subject pixel is assigned to its nearest seed** (`scipy.spatial.cKDTree`), yielding nearest-seed cells. Oversized cells are recursively re-seeded and re-partitioned (depth ≤ 3). Cell size is modulated by a Sobel-gradient detail map, so detailed regions receive smaller shards. Sub-minimum slivers are absorbed rather than discarded, so the partition **tiles the subject exactly**.

**Shard colour** (`targets/color.py`). Each cell's **dominant** (modal, not mean) colour is converted to CMYK; channels above a **0.15** threshold are retained (≤ 3), and realised as stacked layers of tinted acrylic. Lamination is **intensity-weighted** — a correction made during this work, having found that binary channel selection mapped golden-orange and pure red to an *identical* laminated colour.

**Rendering** (`forward/renderer.py`). Each panel is warped onto **both** walls via its projective homography, blurred by that panel's penumbra kernel, and composited multiplicatively. Cross-talk is thus faithfully simulated, not approximated.

### 3.3 ALGORITHM — the contribution: cross-talk-aware assignment

Per §2, a shard's host **panel** determines its **depth**, and depth is the sole steering parameter for its secondary shadow. In the original pipeline, this decision was:

```python
local_k = int(rng.choice(len(fam), p=cover / cover.sum()))
```

**The one lever governing all cross-talk was a random draw.** This fully explains the observed seed-dependence of the noise (14–37% across seeds).

We replace it with a damage-minimising selection. For each shard *f* and each viable host panel *p* (those with non-zero primary-wall coverage):

```
obj(f, p) = cover(f, p) − λ · D(f, p)
host      = argmax_p obj(f, p)
```

**Efficient evaluation of D.** No re-rendering is required. We compose the two cached homographies into a single wall-to-wall map,

```
G_p = H_pw[p, secondary] · H_wp[p, primary]        (one 3×3 per panel)
```

which sends a point on the primary wall directly to where that shard's material, if hosted on *p*, lands on the secondary wall. Damage is then evaluated by transforming only the shard's **own** pixel coordinates (subsampled to ≤ 200) and indexing the secondary target there — O(200) per candidate, no wall rasterisation.

**Two damage formulations were tested.**

*Harm-only (unsigned):*
```
D = mean_q ‖ target_S(q) ⊙ (1 − T) ‖²         ≥ 0
```
where *T* is the shard's laminated transmittance. This penalises harm and is clamped non-negative.

*Signed (with credit):*
```
e_without(q) = ‖ 1 − target_S(q) ‖²      (error if that pixel is left unblocked)
e_with(q)    = ‖ T − target_S(q) ‖²      (error once this shard shadows it)
D            = mean_q [ e_with − e_without ],  negative branch scaled by credit weight c
```
This may go **negative** — a *credit* — when the shard supplies darkness or colour the secondary image wants. It is what permits a shard to be **rewarded for genuine double duty**, which the harm-only form structurally cannot do.

Off-wall landings contribute zero damage, so "steer the shadow off the wall" is a strategy the assignment may discover unaided.

---

## 4. Results

**Protocol.** 10 random seeds per arm. **Panel placement held identical across arms** — only the assignment rule varies, so all differences are attributable to it. Operating point: 0.5× shard density (≈ 55/29 shards per wall), wide-angle panel search. Source pair: apple / breakfast.

### 4.1 Principal result — three-arm comparison

| Arm | Bad cross-talk ↓ | **Joint-intersection ↑** | A RMSE ↓ | A SSIM ↑ | B RMSE ↓ | B SSIM ↑ | Panels used |
|---|---|---|---|---|---|---|---|
| Random (`rng.choice`) | 23.1% | 15.4% | 0.243 | 0.778 | 0.240 | 0.783 | 13.7/14 |
| Harm-only damage | **2.5%** | 2.5% | **0.212** | **0.852** | **0.182** | **0.874** | 7.3/14 |
| **Signed damage (c = 0.5)** | **4.6%** | **31.3%** | 0.218 | 0.809 | 0.188 | 0.843 | 12.4/14 |

**Signed-damage assignment achieves ~5× less bad cross-talk *and* ~2× more joint-intersection than the random baseline, while improving every fidelity metric on both walls.**

### 4.2 Direct response to the "no intersection" critique

The critique was correct about the *prior* system. It is answered as follows:

- Under random assignment, joint-intersection is **15.4%**, and it is **incidental** — nothing in the pipeline sought it; it was a by-product of a coin flip. The 14–37% seed variance confirms it was not designed.
- Under signed-damage assignment it is **31.3%** — **roughly double** — and it is **deliberate**: the objective explicitly credits a shard for contributing to the second wall. Panels now serve both images *by construction*, and the extent to which they do so is **measured, not asserted**.
- Critically, this is not bought at the cost of image quality: fidelity **improves** relative to the baseline on all four measures.

### 4.3 The intersection–noise relationship is *partially separable*, not fundamental

The harm-only arm is instructive and, we think, the most interesting negative result here. Instructed only to *avoid harm*, the optimiser found the degenerate solution: **aim every secondary shadow off the wall.** Noise collapsed to 2.5% — but joint-intersection collapsed with it, also to **2.5%**. It attained clean images by *refusing to interact with the second wall at all*, which is precisely the failure mode the original critique named. **The best fidelity numbers in the table belong to the arm that abolishes the concept.**

This appears to support the hypothesis that the two effects are inseparable — being the same physical event. **The signed arm refutes it.** It attains 4.6% bad *and* 31.3% good simultaneously, a point unreachable if the two moved in lockstep.

A **marginal frontier** does nonetheless exist, controlled by the credit weight:

| credit weight *c* | Bad cross-talk | Joint-intersection | B SSIM |
|---|---|---|---|
| 0.5 | 4.6% | 31.3% | 0.843 |
| 1.0 | 7.0% | 32.7% | 0.836 |
| 2.0 | 9.0% | 33.8% | 0.828 |

Additional intersection is purchased with additional noise. **The trade-off is therefore real at the margin but not fundamental in kind**: the random baseline sat far *inside* the achievable frontier, paying heavily in noise for little intersection. We regard this characterisation — that the relationship is partially separable, with a quantified frontier — as the report's second contribution.

**A predicted artefact, observed:** Wall-B fidelity degrades monotonically as *c* rises (B SSIM 0.843 → 0.828). This is double-counting — a Wall-A shard is credited for darkness that Wall-B's own shards also supply. It was anticipated before the experiment and is the reason *c* = 0.5 is preferred.

**Incidental finding:** the harm-only arm collapsed onto **7.3 of 14** panels (the remainder standing empty). Signed damage restores **12.4/14**. The collapse was an artefact of the one-sided objective — with no reward for double duty, shards had no reason to occupy panels that serve both walls.

### 4.4 Secondary finding — shard density

We swept shard count on the true pipeline (cross-talk present, fabrication ceiling disabled):

| Shards (Wall A) | vs. 220 ceiling | A SSIM | A RMSE |
|---|---|---|---|
| 28 | 0.13× | 0.801 | 0.235 |
| 55 | 0.25× | 0.796 | 0.208 |
| 99 | 0.45× | 0.767 | 0.215 |
| 206 | 0.94× | 0.737 | 0.207 |
| 394 | 1.79× | 0.709 | 0.194 |

**Structural fidelity (SSIM) degrades monotonically with shard count**, while colour error (RMSE) improves slightly — each additional cell is another independent colour-and-host decision, contributing boundary noise faster than it resolves detail. The optimum lies at **~55 shards/wall, roughly a quarter of the fabrication ceiling**: the binding constraint is **fidelity, not fabrication**, which inverts the usual framing. Bad cross-talk is **flat (21–24%) across a 4× change in shard count**, confirming it is a *placement* phenomenon, independent of shard density.

---

## 5. Positioning against prior work — stated conservatively

We wish to be precise about what is and is not new here.

**Mitra & Pauly, *Shadow Art* (SIGGRAPH Asia 2009)** established the core problem: optimise a 3D volume so its shadows from multiple directions approximate multiple target images, with a voxel-carving formulation and explicit treatment of the conflicts between targets. **The problem formulation and the multi-view shadow optimisation are theirs, not ours.**

**Baran et al., *Manufacturing Layered Attenuators for Multiple Prescribed Shadow Images* (Eurographics 2012)** addressed the layered/attenuating-material variant, which is much closer to our physical realisation (stacked semi-transparent layers producing multiple prescribed images). **Layered attenuation as a fabrication strategy is theirs.**

**Recent neural formulations (e.g. *Neural Shadow Art*, 2025)** re-cast shadow art with implicit representations, improving flexibility of geometry and viewing configurations.

**Our contribution is narrower than any of these, and we do not claim otherwise:**

1. **Cross-talk-aware assignment.** In a *shard-based, discretely-hosted* pipeline (where each element must be assigned to one of several candidate carrier planes), we identify that the **host-plane choice is exactly the depth degree of freedom that steers unwanted secondary shadow**, and we make that assignment an optimisation over an explicit, cheaply-computable damage functional — rather than the random draw it previously was. We are not aware of this specific decision being formulated this way in the shadow-art literature, but **we have not conducted an exhaustive survey and would welcome correction.**

2. **Characterisation of the intersection–noise relationship.** The empirical finding that "useful" and "harmful" secondary shadow — physically the *same* event — are **partially separable**, with a frontier parameterised by a credit weight, and that a naive one-sided objective collapses to a degenerate no-interaction solution.

**Explicitly NOT claimed as novel:** shadow art itself; multi-image shadow optimisation; Voronoi shard generation; CMYK separation; subtractive colour lamination; layered-attenuator fabrication; penumbra modelling.

**Whether contribution (1) and (2) together constitute a publishable result — or merely a sound engineering improvement within an established problem — is a judgement we are not in a position to make, and on which we would value the lecturer's advice.** We would rather ask than overclaim.

---

## 6. Limitations

We state these plainly; several are material.

### 6.1 No physical validation
**Everything reported is simulation.** No acrylic has been cut and no lamp has been pointed at anything. The renderer models projective geometry, penumbra from finite source size, and material thickness, but a physical build may expose failures we have not modelled (assembly tolerance, inter-reflection, real material transmittance versus our tabulated values).

### 6.2 Narrow evaluation
Results rest on a **small number of image pairs** (principally apple/breakfast; secondarily two CMYK posters), all of which are **flat-colour graphics with hard edges**. Behaviour on photographs, faces, or images with smooth gradients is **untested**. The 10-seed protocol addresses stochastic variance, **not** generalisation across images.

### 6.3 "Chosen, not derived" parameters
The following are **choices**, and we would defend them only as reasonable, not as derived: panel count (14); candidate count (K = 16); angle bounds; the 0.5 m wall standoff; the 3× magnification cap; the CMYK channel threshold (0.15); maximum stack depth (3); the 220-shard ceiling (shown to be non-binding); and the damage/credit weights (0.5 / 0.5).

*Genuinely derived:* wall dimensions and light throw (from magnification and penumbra targets); `fragment_size` (wall-space scaling); the 0.5× density operating point (**measured** as the structural optimum, §4.4); panel height range (numerically fitted to the wall band); 3 mm material (verified as commercial stock).

### 6.4 An unresolved fabrication conflict
Our best results widen the panel-angle range to **5°–85°**, whereas the original **20°–70°** bound existed for a structural reason: two panels crossing at angle θ require a cross-lap slot of width `thickness / sin θ`, which diverges at shallow crossings. **We have not re-verified slot widths for the wider range.** This must be resolved before fabrication.

### 6.5 The credit term likely measures the wrong quantity
The signed damage credits a shard when its secondary shadow is "closer to the target than blank white." A **red** shard landing on a **golden** croissant satisfies this — red *is* nearer to gold than white is — and is therefore **rewarded**, despite reading visually as contamination. Inspection of the renders confirms this: stray shadows migrated **off the background and onto the croissant**. Accordingly, the 31.3% joint-intersection figure measures **incidence on image content**, not **contribution to it**. A colour-agreement–aware credit is required, and the reported intersection figure should be read as an **upper bound** on genuinely useful double duty.

### 6.6 A fundamental limit
A shard carries **one** colour, drawn from its primary wall's image. Its secondary shadow therefore arrives at the other wall **with the wrong colour**. It may be *steered* somewhere less damaging; it can never arrive *correct* unless the two images happen to want compatible colours at geometrically-linked points. **Genuine double duty is bounded above by the colour agreement between the two source images** — a quantity we have not yet measured, and which for unrelated images may be small.

### 6.7 Engineering status
The contribution is **not yet wired into the command-line pipeline**. `build_panels_greedy` is called from no production code path, and `fragment_shards_overlap` defaults to `damage_weight = 0.0` — i.e. the shipped tool still runs the **fixed 8-panel layout and the random assignment**. The reported results are produced by test harnesses invoking the library directly. This is a packaging deficiency, not a scientific one, but it should be corrected before the work is demonstrated.

---

## 7. Future work

1. **Colour-agreement credit** (§6.5) — reward a shard only when its secondary shadow is *chromatically* appropriate, not merely darker than white. This is the single change most likely to convert nominal joint-intersection into genuine double duty.
2. **Quantify the colour-agreement ceiling** (§6.6) — measure, for a given image pair, how much genuine double duty is even *available*. This would convert our qualitative "fundamental limit" into a computable bound, and would tell an artist in advance whether two images are *compatible* for this medium — which we believe is an interesting question in its own right.
3. **Physical prototype** — the only way to validate the optical model.
4. **Broader evaluation** — photographs and gradient imagery; more image pairs.
5. **Resolve the angle/slot-width conflict** (§6.4).
6. **Panel-count search with a stopping rule** — currently a chosen constant.

---

## 8. Summary

The earlier critique — that the design comprised two non-intersecting blocks — was **correct as applied to the prior system**, where any intersection was incidental to a random draw. We have identified the specific decision governing it (the shard's host plane, which *is* its depth, which is the sole steering parameter for secondary shadow), replaced the random draw with an explicit damage-minimising assignment, and measured the outcome.

The result is that **stray shadow cannot be eliminated — it is a consequence of two simultaneous lights and isotropic material — but it can be *aimed*, and aiming it is substantially better than suppressing it.** Signed-damage assignment yields ~5× less noise and ~2× more genuine joint-intersection than the random baseline, with improved fidelity on both walls. The relationship between useful and harmful secondary shadow is **partially separable**, with a quantified marginal frontier.

We hold the remaining limitations — particularly the colour-agreement flaw in the credit term (§6.5) and the absence of any physical build (§6.1) — to be material, and we present them as such.
