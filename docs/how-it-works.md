# How the Pearl Girl triptych works

Two self-contained explanations of the same system, written for two different readers.

- [Part 1 — for a non-CS reader](#part-1--for-a-non-cs-reader): what the object is and how the
  computer decides where every piece goes. No equations.
- [Part 2 — for a CS reader](#part-2--for-a-cs-reader): formal problem statement, algorithms,
  complexity, and the design decisions with their evidence.

The shipped configuration is arm `30v6`: six sheets, 30 × 30 cm, clear acrylic, four tones.
Numbers throughout are the measured values for that build.

---

# Part 1 — for a non-CS reader

## What the object is

Six flat sheets of clear acrylic, each 29.6 cm square and 3 mm thick, slotted into each other
so they stand up as one rigid cluster about the size of a dinner plate. Every sheet has been
laser-engraved: the laser roughens the surface in patches, and a roughened patch scatters light
instead of passing it, so it reads as grey. We use three engraving depths plus untouched clear
acrylic, giving four tones — near-clear, light grey, mid grey, dark grey.

From any ordinary angle the cluster is visual noise. A few thousand irregular grey patches
floating at different depths, meaning nothing.

Then you put a small bright lamp behind it and it throws a shadow on the wall two and a half
metres away. The shadow is a portrait of the Girl with a Pearl Earring, 1.8 m tall.

Rotate the cluster 120° and the shadow becomes the same girl from the side. Rotate another
120° and it becomes the back of her head. Three portraits, one object, one lamp.

## Why this is hard

The obvious approach — engrave portrait one on sheet one, portrait two on sheet two — does not
work, and it is worth being clear about why, because everything else in the design follows from
this.

**The shadows are not separable.** Light travels in straight lines and does not care which
picture a sheet was meant to serve. When the lamp is lit for the front view, *every* sheet is
in the beam, including the two intended for the side view and the two intended for the back.
Their shadows land on the wall too, at odd stretched angles, smeared across the face. We call
this cross-talk. You cannot switch a sheet off.

**Shadows multiply, they don't add.** Put two half-dark patches in front of one another and you
do not get "fully dark". Each passes half the light, so together they pass a quarter — you get
three-quarters dark. This means a sheet's contribution depends on what every other sheet is
doing at that exact spot, and the interaction is not a simple sum. It is why the naive design
comes out looking like a black blob: three sheets that each wanted to be mid-grey compose to
almost black.

**Any single sheet must be unreadable.** If one sheet carried a recognisable face, the piece
would just be a picture on plastic with decoration around it. The image has to exist only in
the combination. So the picture is deliberately shattered into thousands of fragments and those
fragments are scattered across different sheets at different depths.

## The idea that makes it work

Instead of fighting cross-talk, we exploit it.

A patch of grey sitting in the beam casts a shadow on the wall for *every* view, not just its
own. Usually that is damage. But if the patch is positioned so that its stray shadow happens to
land somewhere the *other* portrait also wanted to be dark, then that stray shadow is not
damage — it is free work. One piece of engraving, three pictures served.

So the real question the program answers is not "where do I put the pieces for the front view".
It is: **where can I put a piece so that it helps all three views at once?**

In the shipped build, **93% of the fragments contribute to all three portraits**, and on
average each one serves 2.93 of the 3 views. The measured cross-talk is *negative* — meaning
the interference between views ended up making the pictures slightly better than if the views
had been solved in isolation. The noise became signal.

## How the computer decides

Six steps.

**1. Read the pictures.** Each of the three portraits becomes a grid of brightness values —
"this spot on the wall should be dark, this spot should be light."

**2. Work out the optics.** Given the lamp position, the sheet positions, and the wall, the
program computes exactly where the shadow of any point on any sheet lands. It also computes how
blurry that shadow will be. This matters: a real lamp is not a mathematical point, it has
width, so shadows have soft edges. A sheet close to the lamp casts a big, blurry shadow; a
sheet further away casts a sharper one. There is no point engraving detail finer than the blur,
so the program measures the blur first and refuses to draw anything smaller.

**3. Shatter the pictures.** Each portrait is broken into irregular organic fragments — think
crazy paving. Not a uniform grid: the program puts *more, smaller* fragments where the picture
has detail (eyes, the edge of the headscarf, the pearl) and fewer, larger ones across flat
areas like the background. The fragments are irregular on purpose, so no sheet shows a
tell-tale grid.

**4. Decide which sheet each fragment goes on.** This is the heart of it, and it is where the
piece stops being a craft project and becomes an optimisation problem.

Each fragment's position *within* its own portrait is already fixed. But which of the sheets
hosts it is still free — and that choice is what decides where its unavoidable stray shadow
lands on the *other* two walls. So the program tries each candidate sheet and asks: if I put
the fragment here, where does its stray shadow fall on the other two portraits, and does it
help or hurt them there? It picks the sheet with the best answer.

The clever part is that it answers this without redrawing the picture. It only transforms the
fragment's own few hundred points, which is thousands of times faster, which is what makes it
possible to do this for every fragment rather than guessing.

**5. Fix the collisions.** The sheets physically cross each other. Where two sheets pass
through the same space, we cut a notch in each so they slide together, like the cardboard
dividers in a wine box. The program finds all such crossings (there are 12) and cuts the
notches. It also checks nothing impossible happened — no three sheets meeting at one point, no
crossing so shallow that the notch would cut the sheet in half.

**6. Fix the tones.** Now the geometry is frozen and only the greys are adjusted.

This step exists because of a real failure. The first renders scored well on the standard
measure and still looked wrong — the outlines were right and the faces were solid black. The
measure we were using only grades the *outline* of a shape, so it was blind to a correctly
shaped, completely crushed face. The cause was the multiplication problem: each fragment picked
its grey based on its own portrait alone, before knowing what would end up stacked behind it.

So the last step goes back over the sheets one at a time, tries all four available tones for
every patch, and keeps whichever combination makes all three portraits closest to correct. It
repeats until nothing improves. On the shipped build this took the accuracy from 0.768 to
0.802, and — more importantly — it is what un-crushed the faces.

## Choosing the design

We did not pick the six-sheet layout by taste. For each design choice, the program built the
entire piece from scratch several times with different settings and measured the result.

Two things we had previously proven turned out to be **wrong** at six sheets:

- With eighteen sheets, *dark* engraving tones won clearly. At six sheets they came **sixth out
  of seven** — light tones won. The reason is the multiplication again: with eighteen layers
  stacking up you need each one pale, with six you don't, and the dark alphabet overshoots.
- With eighteen sheets, spacing the sheets closely was better. At six, wide spacing won.

Both old rules were measured honestly. Both were measured while another setting was pinned by a
constraint that no longer applied. **A measured rule is only valid over the range it was
measured on** — that is probably the most useful thing this project taught us.

## What it costs to use six sheets

We built the eighteen-sheet version too, so the price is known rather than assumed:

| | 6 sheets | 18 sheets |
|---|---|---|
| accuracy (average of 3 views) | 0.846 | 0.882 |
| accuracy (worst single view) | 0.836 | 0.871 |
| cut files to produce | 13 | 37 |
| joints to assemble | 12 | 108 |

Six sheets costs about 4% accuracy and removes 96 assembly joints. Given that every joint is a
chance to misalign the whole piece, that is a trade we would make again.

## Checking it before cutting

Three automatic checks, all of which have to pass before any file is released.

- **Can it be built?** Every joint checked for crossing angle, notch width against material
  thickness, and impossible triple-junctions.
- **Does the cuttable version still work?** The design is simulated once as an idealised
  picture and again as the actual polygon outlines that will be cut, and the two are compared.
  Converting to cuttable shapes changes the result by less than 0.2%.
- **Is every sheet doing real work?** Each sheet is deleted in turn and the piece re-measured.
  If removing a sheet changes nothing, it was decoration. All six matter; removing the least
  important still costs 5%, the most important costs 17%.

## The honest limitations

- **There is no spare capacity.** Six sheets is the minimum that works. Every one is
  load-bearing. Lose or mis-cut a sheet and the piece does not degrade gracefully — it fails.
- **The engraving must be calibrated.** The three greys are specified as *how much light gets
  through*, not as laser power settings, because those differ per machine and per batch of
  acrylic. Whoever cuts this must engrave a test strip and match the tones. This is the single
  most likely way to get a disappointing result from correct files.
- **The lamp must be small.** The whole design assumes a near-point light source. A big soft
  lamp will blur the shadows past the point where the detail exists, and the portraits will
  turn to mush. Position matters too — the geometry is specified to the millimetre.

---

# Part 2 — for a CS reader

## 1. Problem statement

Let $W = \{w_1, w_2, w_3\}$ be three viewing configurations of a turntable. Each $w$ is a
(wall plane, point light) pair; rotating the assembly by $+\theta$ is realised as rotating the
rig by $-\theta$, so the three stops at $\theta \in \{15°, 135°, 255°\}$ become three static
scenes over one fixed object.

The object is a set of $P = 6$ coplanar-grouped rigid sheets. Sheet $p$ occupies a plane at
angle $\phi_p$ with in-plane offset $o_p$, and carries a spatially varying transmittance field
$\alpha_p : [0,1]^2 \to \mathcal{T}$, where $\mathcal{T} = \{1.00, 0.85, 0.62, 0.35\}$ is the
four-level engrave alphabet (clear plus three burn depths).

Given target darkness fields $t_w$, minimise

$$
\mathcal{L}(\{\alpha_p\}) \;=\; \sum_{w \in W} \big\| D_w(\{\alpha_p\}) - t_w \big\|^2
$$

subject to fabricability constraints (minimum feature size, half-lap joint feasibility,
footprint $\le 30$ cm), where the forward model is **multiplicative**:

$$
D_w \;=\; 1 - \prod_{p=1}^{P} \big(1 - \alpha_p \circ H_{pw}\big) * G_{\sigma_{pw}}
$$

with $H_{pw}$ the sheet→wall homography and $G_{\sigma}$ the penumbra PSF.

Three properties make this hard:

1. **No occlusion masking is available.** Every sheet is in every beam. There is no subset
   selection per view; the sum in $\mathcal{L}$ couples all $P$ fields across all $|W|$ views
   simultaneously.
2. **The composition is multiplicative, not additive.** $\mathcal{L}$ is non-convex in
   $\{\alpha_p\}$. Superposition arguments do not apply, and the naive per-view solution
   composes to $\prod \alpha \ll \alpha$ — systematically over-dark.
3. **The output alphabet is discrete** ($|\mathcal{T}| = 4$) and the geometry is a
   *partition* problem, so the search space is combinatorial, not a smooth manifold.

## 2. Forward model

### 2.1 Projection

A point light $L$, a planar sheet, and a planar wall define a central projection. The shadow of
sheet point $P$ is the ray–plane intersection

$$
X = L + t\,(P - L), \qquad t = \frac{n \cdot (p_0 - L)}{n \cdot (P - L)}.
$$

Projection between two planes through a single centre is a projective collineation, hence
exactly representable as a $3\times3$ homography in homogeneous coordinates. We construct it by
projecting the sheet's four corners and solving via **normalised DLT** — isotropic
normalisation for conditioning, then the right null vector of the $2n \times 9$ design matrix by
SVD. Both $H_{pw}$ and its inverse are cached per (sheet, wall) pair.

Magnification is the ray parameter at the sheet centre, $m = d(L, \text{wall}) / d(L,
\text{sheet}) = 6.085$ for the shipped build, mapping a 29.6 cm sheet to a 1.80 m image.

### 2.2 Penumbra as a resolution bound

A finite source of radius $r$ produces a blur whose half-width on the wall scales as
$r \cdot d(\text{sheet}, \text{wall}) / d(L, \text{sheet})$, modelled as a Gaussian $\sigma_{pw}$.
This is not merely a rendering nicety — it is a **hard information-theoretic bound on
achievable detail per depth plane**, and it is fed back into the decomposition as a minimum
feature size. Fragments finer than the worst-case penumbra in a family are merged rather than
fabricated, on the grounds that they are illusory precision. Measured penumbra for the shipped
build is 3.59–3.93 mm at the wall.

### 2.3 Compositing

Independent occluders on a common ray multiply their transmittances. The renderer is
differentiable (PyTorch): `grid_sample` bilinear warp per (sheet, wall), separable 1-D Gaussian
convolution for penumbra, then $\prod_p (1 - \alpha_p)$. Every sheet is warped onto **every**
wall, so cross-talk is not a separate model — it emerges from the same code path.

This single renderer is simultaneously the physics simulator, the optimisation objective, and
the preview. That identity is deliberate: it removes the class of bug where the thing being
optimised diverges from the thing being measured.

## 3. Algorithm

The solver is a **two-stage decomposition**: a combinatorial geometry stage that fixes
*where* material goes, and a coordinate-descent tone stage that fixes *how dark* it is with
geometry frozen. Splitting them is what makes the problem tractable; §3.4 explains why.

### 3.1 Stage 0 — detail-adaptive fragmentation

Per view, the target's subject mask is tiled by **jittered blue-noise seeds → nearest-seed
Voronoi cells**. Seed density is modulated by an importance map (gradient magnitude and colour
transition energy), so fragment size tracks local detail. Post-processing:

- cells exceeding `fragment_max_area` are recursively re-seeded and split;
- cells below the min-area / penumbra floor are dropped;
- every orphaned pixel is reassigned to its nearest surviving cell by **distance transform**,
  guaranteeing hole-free coverage.

The shard budget is enforced as a **ceiling, not a target**. This is a measured result and it
is counter-intuitive: deliberately increasing fragment count *lowers* SSIM and edge fidelity
even as RMSE improves marginally. Each additional fragment is another independent (dominant
tone, host sheet) decision, and those decisions inject boundary noise faster than the finer
tessellation resolves real detail. So the natural detail-biased count is used when it is under
budget, and spacing is coarsened only if it would otherwise be exceeded.

### 3.2 Stage 1 — cross-talk-aware host assignment (the core)

For each fragment $f$ of view $w$, the free variable is the host sheet $p$. Its landing on its
*own* wall is fixed by $H_{pw}$ regardless of host; the host choice determines **only** where
its unavoidable stray shadow lands on the other two walls. That single degree of freedom is the
entire lever on cross-talk.

Originally this was `rng.choice`. That is why cross-talk quality swung 14–37% purely with the
RNG seed.

The replacement is a greedy minimising summed stray damage over all other views:

$$
p^\star(f) = \arg\min_{p \in \text{viable}(f)} \; \sum_{v \neq w} \mathcal{D}(f, p, v)
$$

**The key trick is the damage estimator.** Naively, evaluating a candidate host means
re-rendering a wall — $O(N)$ in wall pixels, per fragment, per candidate, per view. Instead we
precompute a single composed wall-to-wall homography per (sheet, view pair):

$$
G = H_{pv} \, H_{wp}
$$

which maps *primary-wall coordinates directly to secondary-wall coordinates* for material
hosted on sheet $p$. Damage is then evaluated by transforming only the fragment's own pixel
coordinates — subsampled to at most 200 — with **no wall render at all**.

The damage functional itself went through three revisions, and the failures are instructive:

- **v1, unsigned harm.** $\mathcal{D} = \overline{\| t_v(q) \cdot (1 - T_f) \|^2}$ — the light
  the stray shadow steals that view $v$ wanted to keep. Correct behaviour on the easy cases
  (stray on white background → maximal penalty; stray falling off the wall entirely → zero, so
  "steer cross-talk off the wall" is discoverable). But it drives *all* cross-talk to zero,
  including helpful cross-talk, because helping was never worth anything.
- **v2, signed.** $\mathcal{D} = \|T_f - t_v\|^2 - \|1 - t_v\|^2$, allowed to go negative. Now a
  fragment can be *rewarded* for genuine double duty. But raw distance-from-white over-credits:
  any dark fragment on any dark target scores as helpful merely for being darker than blank.
- **v3, gated credit (shipped).** The negative branch is gated on genuine colour agreement,
  $\|T_f - t_v\| < \tau$, using the identical test that the reported metric uses. Measured, v2
  scored ~27% "helpful" cross-talk when the honest colour-agreeing fraction was ~3% — it was
  counting *landed on content*, not *supplied the right value*.

There is also an outline-protection term. The damage functional is edge-blind: a stray shadow
landing on a dark contour steals almost no light, so it scores harmless and can even earn
credit — which is precisely what allows cross-talk to accumulate on the contour carrying the
image. A contour weight map amplifies harm and suppresses credit there.

Result on the shipped build: **100% of fragments serve ≥2 views, 93% serve all 3** (mean 2.93),
and measured cross-talk is **−0.005** — i.e. net *constructive*. The interference term is
negative.

### 3.3 Stage 1b — weave collision resolution

Sheets in different families physically intersect. At each crossing a half-lap slot of depth
`thickness/2` is cut into each. Slot width is $\text{thickness} / \sin\theta$ for crossing
angle $\theta$ — 3.46 mm for 3 mm material at 60°, which is correct and routinely misread as a
bug. Where both sheets hold material at a crossing, a clearance band is trimmed from one.

### 3.4 Stage 2 — tone re-quantisation by coordinate descent

Stage 1 assigns each fragment a tone from *its own view's target*, before anything is known
about what else will occupy the same beam. With 93% of fragments serving all three views, a
typical wall pixel receives its own fragment plus stray from ~2 others, and transmittances
multiply: three sheets each independently wanting 0.6 compose to $0.6^3 = 0.22$. Mid-grey skin
arrives as black.

**This was invisible to IoU.** IoU scores silhouette overlap, so a correctly shaped, uniformly
crushed face scores well. The renders hit IoU 0.82 and looked wrong, and both facts were true.
The lesson generalises: *IoU is a silhouette metric; never report it unpaired.* We now carry
SSIM, RMSE and an edge-fidelity term alongside it, and the failure was found by SSIM.

The first fix attempt was **radiometric pre-compensation** — divide the measured stray light out
of the target and re-solve, standard practice in projector-camera systems. It **oscillated**:
0.690 → 0.598 → 0.665 → 0.641. The reason is structural: re-solving changes which fragments
exist, which changes the stray field, which changes the target. It is a fixed-point iteration
with no contraction guarantee, made worse by the fact that most of the "stray" is deliberate
sharing being double-counted as error.

The shipped fix freezes the geometry entirely. Fragment footprints and host assignments are
fixed; the only free variables are the discrete tone labels. That converts an ill-posed fixed
point into plain **coordinate descent** over $\mathcal{T}^{|\text{regions}|}$: sweep one sheet
at a time, evaluate all four tones, keep the best per pixel. Each sweep is monotone in
$\mathcal{L}$ by construction because a sweep only accepts a decrease.

One refinement matters. MSE is used to *propose* (it is smooth and differentiable and gives a
usable gradient direction), but acceptance is judged on the composite objective. Optimising MSE
directly to convergence produces flat, low-contrast solutions that score well and look worse.
**MSE is the right thing to differentiate and the wrong thing to accept on.**

Measured descent, shipped build:

```
0.7684 → 0.7980 → 0.8019 → 0.8021 → 0.8021 → 0.8021
sheets accepted   5, 5, 3, 2, 1
pixels changed    190057, 46816, 6646, 1498, 5
```

Clean geometric convergence — each sweep touches roughly a quarter of the previous sweep's
pixels.

### 3.5 Stage 3 — vectorisation and fabrication

1. Threshold each sheet's field per tone level.
2. **Enforce minimum feature**: morphological opening (delete islands below cuttable size) then
   closing (delete gaps below cuttable size), structuring element radius `min_feature/2`. This
   is where the penumbra bound becomes physical geometry.
3. Contour → Shapely polygons in sheet-local metres.
4. **Kerf offset** via pyclipper: dilate each ring by half the cut width so the finished part is
   nominal after the beam removes material.
5. Emit DXF/SVG with layers `ENG_L`, `ENG_D`, `ENG_K`, `CUT_SLOT`, `CUT_OUTLINE` — engrave
   before cut, because cutting the outline first leaves a compliant carrier that lifts and
   defocuses.

## 4. Complexity

Let $N$ = wall pixels ($600^2$), $F$ = fragments, $P$ = sheets, $V$ = views, $S = 200$ = damage
subsample cap, $|\mathcal{T}| = 4$.

| stage | cost | note |
|---|---|---|
| homography build | $O(P \cdot V)$ | one SVD each, cached |
| fragmentation | $O(N \log F)$ | nearest-seed + distance transform |
| **host assignment** | $O(F \cdot P \cdot V \cdot S)$ | **independent of $N$** — the whole point |
| naive host assignment | $O(F \cdot P \cdot V \cdot N)$ | what the composed homography avoids |
| collision resolution | $O(P^2)$ | 12 crossings at $P=6$ |
| retone | $O(\text{sweeps} \cdot P \cdot \vert\mathcal{T}\vert \cdot N)$ | 120 renders total |
| vectorise | $O(P \cdot N)$ | morphology + contour |

With $S = 200$ and $N = 360{,}000$, the composed-homography trick is a ~1800× reduction on the
dominant term. That is the difference between "choose hosts by optimisation" and "choose hosts
by coin flip", and the coin flip is what the 14–37% seed variance was.

End-to-end runtime: **21.6 s on an RTX 3060 Ti** for a complete build.

## 5. Search over the design space

The above solves for a *fixed* configuration. The configuration itself — sheet count, pitch,
tone alphabet, fragment density, damage weight — was searched.

Ranking objective, deliberately weighted away from IoU for the reasons in §3.4:

$$
\text{score} = 0.25 \cdot \overline{\text{IoU}} + 0.25 \cdot \min_w \text{IoU}_w + 0.50 \cdot \overline{\text{SSIM}}
$$

The $\min_w$ term is not decoration — it is what prevents the optimiser from sacrificing the
worst view to flatter the mean, which for a three-view piece is a real failure mode.

Runs are memoised and journalled to JSONL, two seeds per configuration, so results survive
interruption and are auditable after the fact.

### The two reversals

We had previously established two rules at eighteen sheets. Re-running the sweeps at six sheets
reversed both.

**Tone alphabet** — `dark_biased` (0.60, 0.30, 0.10) won decisively at 18 sheets. At 6:

| rank | score | alphabet | |
|---|---|---|---|
| 1 | 0.769 | (0.78, 0.60, 0.42) | entered as a weak control |
| 2 | 0.765 | (0.85, 0.62, 0.35) | **shipped** |
| … | | | |
| 6 | 0.742 | (0.60, 0.30, 0.10) | won at 18 sheets |

The mechanism is $\prod(1-\alpha)$ directly: with 18 layers stacking, each must be pale to avoid
saturation. With 6, the dark alphabet overshoots. The old rule was measured correctly — it was
just measured while layer count was pinned by an unrelated constraint.

**Pitch** — tight spacing won at 18 (it had to; 18 sheets in 30 cm leaves no choice). At 6,
wide spacing wins: 0.05 m scores 0.742 against 0.736 at 0.02 m, and cross-talk goes from
$+0.015$ to $-0.002$ — crossing from destructive to constructive.

The generalisable lesson: **a measured lever is valid only over the range it was measured on.**
Both rules were inferred while another variable was pinned by a constraint that no longer
applied. This is a confounding failure, not a measurement failure, and it is not detectable
from the original experiment.

### A negative result worth recording

`damage_weight` produced **identical scores to four decimal places for every non-zero value**
tested (0.02 through 1.0). The cross-talk term is effectively **binary** — it matters that it is
on, not how strongly it is weighted. Its continuous parameterisation is unexercised, which is
worth knowing before anyone spends time tuning it.

## 6. Verification

Three gates, all required to pass.

**Gate A — mechanical.** Crossing count, cluster multiplicity, triple-point detection, minimum
crossing angle, slot width against material thickness. Shipped: 12 crossings, max multiplicity
2, minimum angle 60°, widest slot 3.46 mm on 3.0 mm stock.

**Gate B — fabrication round-trip.** Re-render from the *exported polygons* rather than the
solver's raster and compare. This closes the gap between "what was optimised" and "what will be
cut" — post-morphology, post-kerf. Cost: +0.0018 mean IoU, +0.0105 SSIM.

**Gate C — ablation.** Delete each sheet, re-measure. Shipped: 0/6 idle, drops ranging 0.052 to
0.166.

Gate C caught a genuine near-miss in reasoning. Sheet `F2_0` carries 32 engraved regions against
~990 on its neighbours and looked idle. It has the **largest ablation drop of all six** (0.166
mean, 0.287 on the front view) — re-toning had merged its tones into a few large areas.
**Region count is not contribution.** A gate that only ever passes is not a gate; this one
earned its place.

## 7. Results

```
mean IoU  0.8460     min IoU  0.8358     SSIM  0.7506
RMSE      0.1405     edge fidelity  0.5702     distinctness  0.307
3629 polygons over 6 sheets, 13 cut files
cross-talk  -0.005 (constructive)
sharing: 100% of fragments serve >=2 views, 93% serve all 3 (mean 2.93)
runtime 21.6 s (CUDA)
```

Against the 18-sheet build (0.882 mean / 0.871 min), the six-sheet constraint costs 0.036 mean
IoU and 0.035 on the worst view, in exchange for 96 fewer joints. Since every joint is an
independent alignment error source, and the piece has no redundancy, we regard that as
favourable.

## 8. Open items

- **Vertical streaking.** Fragment boundaries correlate weakly along the vertical axis. Blue-
  noise error diffusion in the tone quantisation is the obvious fix and is not implemented.
- **`lambda_crosstalk` unexercised.** An explicit cross-talk penalty exists in the gradient path
  and has never been swept; per §5 it may well be redundant with the greedy.
- **Greedy is myopic.** Host assignment is single-pass in fragment index order with no lookahead
  and no revisiting. A fragment placed early cannot be moved once later fragments reveal a
  better arrangement. Beam search or a second reassignment pass conditioned on the realised
  stray field is the natural next step.
- **No redundancy.** Six sheets is minimal. This is a property of the artefact, not a bug, but
  it means the piece fails rather than degrades.

---

## Source files

| stage | file |
|---|---|
| configuration and build driver | [pearl3_baseline.py](../pearl3_baseline.py) |
| parameter search | [pearl3_sweep.py](../pearl3_sweep.py) |
| tone coordinate descent | [pearl3_retone.py](../pearl3_retone.py) |
| fabrication and gates | [pearl3_fab.py](../pearl3_fab.py) |
| deliverable packaging | [pearl3_package.py](../pearl3_package.py) |
| fragmentation and host assignment | [shadowart/solve/decompose.py](../shadowart/solve/decompose.py) |
| differentiable renderer | [shadowart/forward/renderer.py](../shadowart/forward/renderer.py) |
| projection and homographies | [shadowart/geometry/](../shadowart/geometry) |
| vectorisation and export | [shadowart/raster2vec/](../shadowart/raster2vec), [shadowart/fabricate/](../shadowart/fabricate) |

Related reading: [pipeline.md](pipeline.md) (general two-wall pipeline),
[30x30-optimization.md](30x30-optimization.md) (the six-sheet decision record),
[crosstalk-noise.md](crosstalk-noise.md), [greedy-partitioning.md](greedy-partitioning.md).
