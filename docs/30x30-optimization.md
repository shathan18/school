# The 30 × 30 cm Pearl Girl triptych — optimisation record

A full re-optimisation of the three-view turning sculpture under the revised brief: a
**30 × 30 cm** footprint (not 60 × 60), **clear Perspex engraved to four tones** (not
alcohol-ink dye), and **no minimum shard size**.

Nothing was carried over from the previous solution. Every number below was re-measured.

> **Source images.** All results use the `pearlN` view set. The previous `girl3` set was
> abandoned because its back view's matte was torn — see §8.

---

## 1. What ships

**Arm `30v6` + stage-2 re-toning.** Files in `out_pearl3_30/v6/`.

The build is **six sheets**, two per family. That is a hard constraint handed down after the
18-sheet build was complete, not a search result — and §3a is the record of re-deriving every
parameter underneath it, because two of the rules found at 18 sheets turned out to be **wrong**
at 6.

|                                              | mean IoU  | worst view | SSIM      | RMSE      | edge fidelity |
| -------------------------------------------- | --------- | ---------- | --------- | --------- | ------------- |
| Previous 60 × 60 solution (18 sheets)        | 0.722     | —          | —         | —         | —             |
| 6 sheets, carrying the 18-sheet parameters   | 0.814     | 0.786      | 0.672     | 0.186     | 0.493         |
| 6 sheets re-optimised (stage 1)              | 0.845     | 0.827      | 0.700     | 0.161     | 0.513         |
| **6 sheets re-optimised + re-toned, as exported** | **0.846** | **0.836** | **0.751** | **0.141** | **0.570**     |

The last row is the one that counts: it is measured by re-rendering **from the exported cut
files**, not from the solver's raster. A quarter of the footprint and a third of the sheets of
the original, and still comfortably ahead of it.

**What the constraint costs.** The 18-sheet build (`30v4`, still in `out_pearl3_30/v4/`)
exports at 0.882 mean / 0.871 worst. Six sheets gives up **0.036 mean IoU and 0.035 on the
worst view** — but re-optimising recovered 0.031 of the 0.040 that the constraint initially
took, so most of the naive loss was tuning, not material.

| view         | IoU   | SSIM  | RMSE  | edge  |
| ------------ | ----- | ----- | ----- | ----- |
| back (0°)    | 0.851 | 0.787 | 0.129 | 0.660 |
| side (120°)  | 0.853 | 0.768 | 0.144 | 0.636 |
| front (240°) | 0.839 | 0.729 | 0.145 | 0.635 |

(Per-view figures are the solver raster; the exported files cost a further +0.002 mean IoU,
§6 gate B.)

### Chosen configuration

| parameter           | value                                                    | why                                       |
| ------------------- | -------------------------------------------------------- | ----------------------------------------- |
| panels              | **6** sheets, 3 families of 2                            | hard build constraint                     |
| panel pitch         | **50 mm**                                                | §3a — reversed from 20 mm                 |
| family angles       | 15° / 75° / 135° → one family per viewing stop           | §2, finding 2                             |
| footprint           | **30 × 30 cm**, swept circle 30.0 cm                     | hard constraint, verified by gate 3       |
| sheet               | 29.9 cm square, 3 mm clear Perspex                       | one pitch gap, so the sheets run near full |
| viewing stops       | 3, at 120° spacing                                       | §5                                        |
| engrave tones       | clear / 0.85 / 0.62 / 0.35 transmittance                 | §3a — reversed from dark-biased           |
| shard granularity   | `fragment_size` 0.05, budget 400                         | §3a                                       |
| lamp                | 0.493 m behind the body, 0.874 m high                    |                                           |
| wall (image)        | 2.507 m in front, 1.80 m image → **6.09× magnification** |                                           |
| engrave min feature | **0.2 mm** (beam spot)                                   | §4                                        |
| cut min feature     | 5 mm                                                     | structural, cut layers only               |

1140 shards, 3629 engraved regions over 6 sheets, 12 crossings, 13 cut files.

---

## 2. Three findings that had to come before any optimisation

**1. The published 0.83 excluded the noise.** The previous figure was a _shards-only_ render.
With all light present the same build scored **0.722**. That 0.108 gap is not an accounting
detail — it _is_ the projection noise the brief asks to reduce. Everything here is reported
all-light.

**2. Two panel families cannot serve three views.** Host assignment is winner-take-all via
`primary_wall_of`, so with two families the third view was served only by whatever fell out.
Fixed by giving each stop its own family at `(270 − θ) mod 180`.

**3. The 30 × 30 deficit was cross-talk, not resolution.** The natural assumption is that a
half-size piece has half the detail. Measured shards-only, 30 × 30 and 60 × 60 were a tie
(0.828 vs 0.830). The entire deficit was stray light. This is what made the target reachable:
there was nothing to recover in resolution, and everything to recover in noise.

---

## 3. What actually moved the number

Each lever was swept independently with ≥2 seeds averaged (`pearl3_sweep.py`).

**Lever 1 — damage-aware host selection: +0.058.** Price each shard by the harm it does on the
views it does _not_ serve. 0.689 → 0.746, cross-talk cost 0.137 → 0.076. It is a **binary
lever**: every `damage_weight` in [0.01, 2.0] scores identically, because 0.0 takes a different
random-host code path. There is nothing to tune. `credit_weight` is monotonically negative;
leave it off.

**Lever 2 — many sheets, tight pitch: +0.070, the largest.**

| sheets/family | pitch     | mean IoU  | worst     | cross-talk cost |
| ------------- | --------- | --------- | --------- | --------------- |
| 2             | 60 mm     | 0.746     | 0.672     | +0.076          |
| 3             | 30 mm     | 0.796     | 0.744     | +0.014          |
| 4             | 20 mm     | 0.810     | 0.759     | −0.014          |
| **6**         | **20 mm** | **0.816** | **0.782** | **−0.033**      |
| 10            | 25 mm     | 0.821     | 0.782     | +0.020          |
| 12            | 15 mm     | 0.812     | 0.781     | −0.050          |

Note the sign change: past ~4 sheets per family the cross-talk cost goes **negative** — the
all-light render beats the shards-only render, i.e. stray light stopped being noise and started
contributing. 18 sheets is the knee; 30 sheets buys 0.005 more and is not worth the material.

**Lever 3 — allocate tones evenly in optical density, not transmittance: +0.034.**
Dark-biased (0.60, 0.30, 0.10) scores 0.819; even-in-transmittance (0.75, 0.50, 0.25) scores
0.785. Because overlapping sheets _multiply_, even spacing in transmittance clusters the
achievable stack tones at the light end and leaves the midtones unreachable.

**Lever 4 — re-tone with the geometry frozen: +0.021 IoU, +0.039 SSIM, −0.033 RMSE.** §4.

### Refuted — measured, negative, do not re-run

| hypothesis                                 | result                                                                |
| ------------------------------------------ | --------------------------------------------------------------------- |
| Higher wall/panel resolution               | flat (600 px ≡ 300 px; GPU is latency-bound, so it was free to check) |
| Larger image / magnification               | flat across the whole range                                           |
| Larger source radius (softer light)        | monotonically worse                                                   |
| `colour_blend`                             | −0.019                                                                |
| `max_stack` 1 vs 2 vs 3                    | **no effect at all** under a 4-tone engrave palette                   |
| Radiometric pre-compensation               | oscillates: 0.690 → 0.598 → 0.665 → 0.641                             |
| Band-limiting the targets                  | monotone loss, no interior optimum (see `prior-art.md` §2)            |
| `fragment_size` 0.05–0.12                  | flat                                                                  |
| Dropping to 15 sheets to shed the idle one | fails the weave gate (19 triple points), −0.021 IoU                   |

---

## 3a. Re-optimising for six sheets — where the 18-sheet rules broke

Six sheets arrived as a build constraint after §3 was finished. The honest thing to do with a
table of levers measured at 18 sheets is to distrust it, because §3's lever 2 says the
cross-talk cost only turns negative past about 12 sheets. Below that knee, stray light is noise
again, and every parameter that was tuned against constructive cross-talk is being asked a
different question. So all of it was re-swept (`pearl3_sweep.py v6*`, ≥2 seeds).

Dropping straight from 18 to 6 with the shipped parameters cost **0.040 mean IoU** (0.854 →
0.814) and flipped the cross-talk cost from −0.042 (helping) to +0.017 (hurting). Re-optimising
recovered 0.031 of that. Two levers reversed outright.

**Reversal 1 — pitch wants to OPEN, not tighten.**

| pitch  | mean IoU  | worst     | cross-talk cost |
| ------ | --------- | --------- | --------------- |
| 20 mm  | 0.815     | 0.785     | +0.015          |
| 30 mm  | 0.822     | 0.797     | +0.006          |
| **50 mm** | **0.830** | **0.816** | **−0.002**   |
| 70 mm  | 0.821     | 0.792     | +0.009          |
| 100 mm | 0.780     | 0.769     | +0.060          |

§3's lever 2 concluded "tighter is better at every sheet count". That was measured where the
30 × 30 footprint solve **capped pitch at 20 mm** — with six sheets per family there are five
gaps to fit inside a 30 cm swept circle. So "tighter is better" was only ever observed on one
side of the optimum. With one gap instead of five there is room to open up, and 50 mm both
recovers the worst view (0.785 → 0.816) and drives cross-talk back to neutral. Past that it
collapses: at 100 mm the outer sheets sit far enough off-axis that their stray shadows land
somewhere unrelated to where they were priced.

**Reversal 2 — the engrave alphabet must go LIGHT.**

| levels             |           | mean IoU  | SSIM      | RMSE      |
| ------------------ | --------- | --------- | --------- | --------- |
| (0.78, 0.60, 0.42) | *shallow* | 0.839     | **0.708** | **0.161** |
| **(0.85, 0.62, 0.35)** | *light-biased* | **0.840** | 0.697 | 0.164 |
| (0.56, 0.32, 0.18) | uniform-D | 0.831     | 0.677     | 0.178     |
| (0.60, 0.30, 0.10) | *dark-biased — shipped at 18* | 0.830 | 0.661 | 0.186 |
| (0.75, 0.50, 0.25) | uniform-T | 0.818     | 0.683     | 0.172     |
| (0.80, 0.40, 0.06) | wide      | 0.820     | 0.644     | 0.192     |

This is the sharper reversal. At 18 sheets dark-biased won and (0.78, 0.60, 0.42) was entered
as a *deliberately weak control arm*; at 6 sheets the control comes **first** and dark-biased
falls to sixth of seven. The probe said why before the sweep did: it over-covered every view
(front foreground 0.43 against a target 0.36). With two sheets per family instead of six there
are far fewer transmittances to multiply, so a dark alphabet has no light one to walk back
with and the picture just goes muddy. Lighter levels restore the ladder — SSIM 0.661 → 0.697,
RMSE 0.186 → 0.164.

(0.85, 0.62, 0.35) ships over the marginally better-scoring (0.78, 0.60, 0.42) on a
**fabrication** argument, not a numerical one: the two are inside seed noise of each other, but
its levels are 23 and 27 points apart instead of 18, so the engraver has room to hold them
apart. `intensity_gain` swept flat over 0.8–1.0 and stays at 1.0.

**Confirmed, not reversed.** `damage_weight` is still perfectly binary — 0.02, 0.1, 0.25, 0.5
and 1.0 give *identical* results to four decimals. Shard granularity is still nearly flat: every
combination of `fragment_size` and `shard_budget` lands within 0.005, against a seed sigma of
0.002. It was taken to the fine end (0.05 / 400 → 1140 shards, ~190 per sheet) because the brief
lifted the minimum shard size, edge fidelity is the one metric that separates the runs
(0.513 against 0.489–0.498 for everything coarser), and with six sheets doing the work of
eighteen, resolution is the only currency left.

**The lesson worth keeping.** Both reversals have the same shape: a rule was inferred inside a
region where one variable was pinned by an unrelated constraint (pitch by the footprint solve,
tone by the number of layers available to multiply), and it did not survive being asked outside
that region. A measured lever is only valid over the range it was measured on.

---

## 4. The two constraints that were capping quality

Both were found by the fabrication gates, and neither was a bug that raised an error.

**The minimum feature size was the wrong constraint.** A stage-2 pass that re-quantises tone
per pixel with the geometry frozen improved every raster metric — and then scored **0.4154**
through the export round trip. `enforce_min_feature` was applying a 5 mm floor to _engraved_
regions. But 5 mm is the limit for a shard **cut out** of a sheet, which must survive as a
self-supporting object. An engraved region is a mark on a sheet that stays whole; its limit is
the beam spot, ~0.2 mm — _under half a panel pixel_. The brief had already said there was no
minimum shard size; the pipeline was still enforcing one inherited from a dyed-and-cut
workflow. The structural limit still governs the cut layers, where it belongs.

**Marching squares is half a pixel small.** Contours are extracted at the 0.5 level, which runs
through the **centres** of the boundary pixels — so every exported region is inset by half a
pixel per side. The error scales with total perimeter, which is exactly why it hid: the same
exporter at the same settings cost +0.0037 mean IoU on stage 1's 281 large regions and +0.0496
on stage 2's 5402 small ones. A half-pixel outward grow brings it to **+0.0011**. This is
registration, not kerf compensation — engraving removes nothing at the boundary.

### The re-toning stage, and the four designs that failed first

Stage 1 chooses each shard's tone from its own view's target _before_ host assignment. But 93%
of shards serve all three views and transmittances multiply, so three sheets each wanting 0.60
compose to 0.216 — interiors crush to black while the silhouette stays perfect. **IoU cannot
see this**, which is why it went unnoticed for so long.

| design                                                   | raster IoU | through the export                                      |
| -------------------------------------------------------- | ---------- | ------------------------------------------------------- |
| (a) per pixel, 5 mm floor                                | 0.8283     | **0.4154** — gain was finer than the export could carry |
| (b) per shard (281 regions)                              | 0.7221     | 0.7078 — fabricable, and worse than doing nothing       |
| (c) per pixel + 5 mm projection each sweep               | 0.1850     | progressive erosion to nothing                          |
| (d) (c) + accept/reject                                  | 0.8079     | descent stalls, 0/18 sheets accepted                    |
| **(e) per pixel + beam-spot projection + accept/reject** | **0.8391** | **0.8380**                                              |

_This comparison was run on the earlier `girl3` target set (§8) and is kept at those numbers
because it is a comparison between designs, not a result. Re-running (a)–(d) on `pearlN` would
move all five rows together and change nothing about which design wins. Design (e) on the
current targets scores **0.8940 raster / 0.8822 exported**._

Two things make (e) work. The constraint applied during the descent is the _real_ one (§4), and
**squared error proposes while the composite objective accepts**: per-pixel MSE is cheap and
gives a good move, but MSE alone will happily trade silhouette coverage for tonal error — that
is precisely how (b) lost 0.10 IoU. So each sheet's proposed map is only kept if
`0.25·mean_IoU + 0.25·min_IoU + 0.50·SSIM` improves, and rolled back otherwise. The descent is
monotone: 16, then 13, 9, 7, 4 sheets accepted over five sweeps. It also **deletes 21% of the
shard material** — a fair amount of the stage-1 solution was actively harmful.

---

## 5. Optimiser bake-off

Five methods, equal budget of 25 distinct evaluations each, shared memoised journal, identical
9-dimensional search space (sheet count, pitch, three tone levels, fragment size, intensity
gain, detail bias, grid phase). `pearl3_search.py`, results in `out_pearl3/search/bakeoff.json`.

| method                | best score | wall time |
| --------------------- | ---------- | --------- |
| **genetic algorithm** | **0.7874** | 577 s     |
| random search         | 0.7838     | 761 s     |
| TPE (Optuna)          | 0.7806     | 451 s     |
| CMA-ES                | 0.7799     | 420 s     |
| simulated annealing   | 0.7487     | 424 s     |
| _hand-tuned `30v4`_   | _0.7794_   | —         |

**The informative result is that random search is within 0.004 of the winner.** At this budget
the response surface has no exploitable structure — which is Bergstra & Bengio's finding
reproduced on this problem, and it is the stopping signal the brief asked for. Continuing to
tune parameters is not where the remaining quality is; the +0.021 from re-toning and the
+0.049 recovered from the registration fix both dwarf the +0.008 the best optimiser found over
a hand-tuned point.

Two useful confirmations fell out of it: every method independently converged to 5–7 sheets per
family at 14–19 mm pitch (lever 2's knee), and every method independently chose dark-biased
tone levels (lever 3).

---

## 6. Fabrication verification

Three gates, all run on the shipped build by `pearl3_fab.py`.

**Gate A — weave feasibility: PASS.** 12 crossings resolve into 12 clusters with maximum
multiplicity 2, so there are no triple points (which cannot be assembled from flat sheets even
though the solver is happy to propose them). Minimum crossing angle 60°. The widest slot is
**3.46 mm for 3.0 mm material**, because a slot at 60° must be `thickness / sin θ` wide — the
joint generator had been cutting fixed-width notches, and those sheets would not have gone
together. Six sheets at 50 mm pitch is a much simpler weave than 18 at 20 mm: 12 crossings
rather than 108, and the triple-point risk that killed the 15-sheet variant does not arise.

**Gate B — export round trip: +0.0018 mean IoU.** Re-renders from the exported polygons using
point-in-polygon on pixel centres. This gate found both problems in §4. The cost is far smaller
than the 18-sheet build's +0.0119 simply because there are 3629 polygons to register rather
than 5583, and this error scales with total perimeter.

**Gate C — single-panel ablation: 0 of 6 sheets idle, and the load splits 2 / 2 / 2 across the
three views.** Blanking each sheet in turn and measuring the IoU each view loses is the only
test of duty that cannot be faked by a shard that merely exists. With only six sheets the test
bites much harder: removing the weakest sheet now costs 0.052 mean IoU and the strongest costs
0.166, against a handful of thousandths when eighteen sheets could cover for each other.

**Region count is not contribution.** Sheet `F2_0` carries just 32 engraved regions against
~700 on its neighbours, which looks like an idle sheet and is not — it has the *largest*
ablation drop of all six (0.166 mean, 0.287 on the front view). Re-toning merged its tones into
a few big contiguous areas. Counting polygons would have condemned the single most important
sheet in the build.

**Removing sheets is not free, and gate A is why.** Before the target fix, the obvious response
to an idle sheet was to drop to five per family. Re-solved at 15 sheets (`30v5`), that build
**fails gate A with 19 triple points** — at 20 mm pitch the three families land on a lattice
where three sheets meet at a common point, which cannot be assembled from flat interlocking
material at all. Ablation says a sheet is _idle_, not that it is _removable_: remove it and the
solver redistributes onto the rest and the geometry changes underneath.

---

## 7. Known limitations, stated plainly

- **Tonal quality still lags silhouette quality.** All three views are now legible as the
  portrait, but SSIM sits at 0.75 against IoU at 0.85 — the outline is much better resolved
  than the shading inside it. IoU measures the silhouette, and the silhouette is not the
  picture; SSIM, RMSE and edge fidelity are reported throughout for exactly this reason.
- **Six sheets costs 0.036 mean IoU against eighteen, and that is real.** §3a recovered most of
  the naive loss but not all of it, and the gap that remains is edge fidelity above all
  (0.570 against 0.630). There is no tuning left to close it — the material is simply not
  there. If the sheet count ever reopens, the frontier in §3 lever 2 says the return runs to
  about 18 and is flat after.
- **There is no redundancy left.** With eighteen sheets, losing one cost thousandths; with six,
  the ablation drops run 0.052 to 0.166. Every sheet is load-bearing, so a mis-cut sheet is not
  a cosmetic problem, and there is no spare capacity to absorb fabrication tolerance.
- **The three views are more alike than before.** Distinctness fell from 0.43 to 0.35 when the
  targets changed, because `pearlN`'s three poses (front, three-quarter, back) share more of
  their mass than `girl3`'s did (front, true profile, back). That makes the reconstruction
  easier and the _illusion_ less startling — a real trade, and one worth revisiting if the
  point of the piece is the surprise rather than the fidelity.
- **Vertical streaking** remains visible in all three views. Frequency-shaped/blue-noise error
  diffusion is the untested idea most likely to help; a naive heavy de-streak was tried
  previously and cost IoU 0.83 → 0.55.
- **Between the stops, the piece is not resolved.** At the three viewing angles it scores
  0.84–0.85; at the intermediate ¾ angles it falls away sharply. A 5-stop variant (`30seq`)
  spreads the quality more evenly but drops the worst view a long way, which is worse where it
  matters most. The 3-stop build was chosen deliberately: a turning sculpture is judged at its
  stops.
- `SolveParams.lambda_crosstalk` exists and has never been exercised, because the partition
  solver does not call the optimiser path that uses it.

---

## 8. The source images, and a result that was not the optimiser's fault

The renders were reproducing large white voids — a bite out of the back view's turban, ragged
gaps across both shoulders. The obvious suspects were the re-toning stage, which deletes
material by design, and the cross-talk. Both were innocent: **the holes were in the source PNG.**
`examples/girl3_back.png` is a botched background removal that ate into the figure, and the
solver was reproducing it faithfully.

`check_targets.py` scores this so it cannot recur silently:

| image               |  holes % | raggedness |
| ------------------- | -------: | ---------: |
| `girl3_back.png`    | **0.00** |   **5.08** |
| `girl3_profile.png` |     0.49 |       4.14 |
| `girl3_front.png`   |     0.48 |       3.22 |
| `pearlN_back.png`   |     0.16 |       2.26 |
| `pearlN_side.png`   |     1.54 |       2.42 |
| `pearlN_front.png`  |     1.98 |       2.80 |

The instructive column is the first one. `girl3_back.png` scores **0% holes** — a
`binary_fill_holes` test, the natural way to look for this, says the image is perfect. The
damage is open to the image border, so it is not an enclosed hole and no hole-fill repairs it.
Only the perimeter-to-area measure catches it, at 5.08 against ~2.3 for a clean cutout. (The
higher holes % on `pearlN` is genuine enclosed background — the gap between the turban tail and
the shoulder — which is why raggedness, not holes, is the discriminator.)

Switching to `pearlN` cost about 40% of the source pixels and was still clearly right:
resolution had already measured **flat** as a lever (§3), and the built piece resolves roughly
55 features across the image — far below either source. What it bought:

|                          | mean IoU  | worst view | idle sheets |
| ------------------------ | --------- | ---------- | ----------- |
| `girl3` (torn back view) | 0.838     | 0.817      | 1 of 18     |
| **`pearlN` (clean)**     | **0.882** | **0.871**  | **0 of 18** |

The idle sheet was a symptom, not a design flaw. A view missing a third of its content cannot
give its panel family enough work to do, so a sheet sat out; with complete targets the load
balances 6 / 6 / 6 and every sheet earns its place. Roughly a third of the total improvement in
this project came from fixing inputs and constraints rather than from optimisation.
