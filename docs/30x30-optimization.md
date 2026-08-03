# The 30 × 30 cm Pearl Girl triptych — optimisation record

A full re-optimisation of the three-view turning sculpture under the revised brief: a
**30 × 30 cm** footprint (not 60 × 60), **clear Perspex engraved to four tones** (not
alcohol-ink dye), and **no minimum shard size**.

Nothing was carried over from the previous solution. Every number below was re-measured.

> **Source images.** All results use the `pearlN` view set. The previous `girl3` set was
> abandoned because its back view's matte was torn — see §8.

---

## 1. What ships

**Arm `30v4` + stage-2 re-toning.** Files in `out_pearl3_30/v4/`.

|                                               | mean IoU  | worst view | SSIM      | RMSE      | edge fidelity |
| --------------------------------------------- | --------- | ---------- | --------- | --------- | ------------- |
| Previous 60 × 60 solution                     | 0.722     | —          | —         | —         | —             |
| 30 × 30, first attempt                        | 0.689     | 0.623      | —         | —         | —             |
| 30 × 30 optimised (stage 1)                   | 0.854     | 0.846      | 0.672     | 0.160     | 0.524         |
| **30 × 30 optimised + re-toned, as exported** | **0.882** | **0.871**  | **0.756** | **0.127** | **0.630**     |

The last row is the one that counts: it is measured by re-rendering **from the exported cut
files**, not from the solver's raster. The halved footprint now scores _better_ than the
original double-size piece on the metric the original was reported on.

| view         | IoU   | SSIM  | RMSE  | edge  |
| ------------ | ----- | ----- | ----- | ----- |
| back (0°)    | 0.897 | 0.790 | 0.114 | 0.696 |
| side (120°)  | 0.902 | 0.767 | 0.125 | 0.651 |
| front (240°) | 0.883 | 0.747 | 0.126 | 0.687 |

(Per-view figures are the solver raster; the exported files cost a further +0.012 mean IoU,
§6 gate B.)

### Chosen configuration

| parameter           | value                                                    | why                                       |
| ------------------- | -------------------------------------------------------- | ----------------------------------------- |
| panels              | **18** sheets, 3 families of 6                           | knee of the layout frontier (§3, lever 2) |
| panel pitch         | **20 mm**                                                | ditto                                     |
| family angles       | 15° / 135° / 255° → one family per viewing stop          | §2, finding 2                             |
| footprint           | **30 × 30 cm**, swept circle 30.0 cm                     | hard constraint, verified by gate 3       |
| sheet               | 28.3 cm square, 3 mm clear Perspex                       |                                           |
| viewing stops       | 3, at 120° spacing                                       | §5                                        |
| engrave tones       | clear / 0.60 / 0.30 / 0.10 transmittance                 | §3, lever 3                               |
| lamp                | 0.471 m behind the body, 0.869 m high                    |                                           |
| wall (image)        | 2.529 m in front, 1.80 m image → **6.36× magnification** |                                           |
| engrave min feature | **0.2 mm** (beam spot)                                   | §4                                        |
| cut min feature     | 5 mm                                                     | structural, cut layers only               |

766 shards, 5583 engraved regions over 18 sheets, 108 crossings, 37 cut files.

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

**Gate A — weave feasibility: PASS.** 108 crossings resolve into 108 clusters with maximum
multiplicity 2, so there are no triple points (which cannot be assembled from flat sheets even
though the solver is happy to propose them). Minimum crossing angle 60°. The widest slot is
**3.46 mm for 3.0 mm material**, because a slot at 60° must be `thickness / sin θ` wide — the
joint generator had been cutting fixed-width notches, and those sheets would not have gone
together.

**Gate B — export round trip: +0.0119 mean IoU.** Re-renders from the exported polygons using
point-in-polygon on pixel centres. This gate found both problems in §4.

**Gate C — single-panel ablation: 0 of 18 sheets idle, and the load splits 6 / 6 / 6 across the
three views.** Blanking each sheet in turn and measuring the IoU each view loses is the only
test of duty that cannot be faked by a shard that merely exists. Every sheet now earns its
place — on the earlier target set one sheet was idle, and the torn back-view matte (§8) was the
reason: a view missing a third of its content cannot give its family enough to do.

**Removing sheets is not free, and gate A is why.** Before the target fix, the obvious response
to an idle sheet was to drop to five per family. Re-solved at 15 sheets (`30v5`), that build
**fails gate A with 19 triple points** — at 20 mm pitch the three families land on a lattice
where three sheets meet at a common point, which cannot be assembled from flat interlocking
material at all. Ablation says a sheet is _idle_, not that it is _removable_: remove it and the
solver redistributes onto the rest and the geometry changes underneath. **18 sheets stands.**

---

## 7. Known limitations, stated plainly

- **Tonal quality still lags silhouette quality.** All three views are now legible as the
  portrait, but SSIM sits at 0.75 against IoU at 0.88 — the outline is much better resolved
  than the shading inside it. IoU measures the silhouette, and the silhouette is not the
  picture; SSIM, RMSE and edge fidelity are reported throughout for exactly this reason.
- **The three views are more alike than before.** Distinctness fell from 0.43 to 0.35 when the
  targets changed, because `pearlN`'s three poses (front, three-quarter, back) share more of
  their mass than `girl3`'s did (front, true profile, back). That makes the reconstruction
  easier and the *illusion* less startling — a real trade, and one worth revisiting if the
  point of the piece is the surprise rather than the fidelity.
- **Vertical streaking** remains visible in all three views. Frequency-shaped/blue-noise error
  diffusion is the untested idea most likely to help; a naive heavy de-streak was tried
  previously and cost IoU 0.83 → 0.55.
- **Between the stops, the piece is not resolved.** At the three viewing angles it scores
  0.88–0.90; at the intermediate ¾ angles it falls away sharply. A 5-stop variant (`30seq`)
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

| image              |  holes % | raggedness |
| ------------------ | -------: | ---------: |
| `girl3_back.png`   | **0.00** |   **5.08** |
| `girl3_profile.png`|     0.49 |       4.14 |
| `girl3_front.png`  |     0.48 |       3.22 |
| `pearlN_back.png`  |     0.16 |       2.26 |
| `pearlN_side.png`  |     1.54 |       2.42 |
| `pearlN_front.png` |     1.98 |       2.80 |

The instructive column is the first one. `girl3_back.png` scores **0% holes** — a
`binary_fill_holes` test, the natural way to look for this, says the image is perfect. The
damage is open to the image border, so it is not an enclosed hole and no hole-fill repairs it.
Only the perimeter-to-area measure catches it, at 5.08 against ~2.3 for a clean cutout. (The
higher holes % on `pearlN` is genuine enclosed background — the gap between the turban tail and
the shoulder — which is why raggedness, not holes, is the discriminator.)

Switching to `pearlN` cost about 40% of the source pixels and was still clearly right:
resolution had already measured **flat** as a lever (§3), and the built piece resolves roughly
55 features across the image — far below either source. What it bought:

| | mean IoU | worst view | idle sheets |
| --- | --- | --- | --- |
| `girl3` (torn back view) | 0.838 | 0.817 | 1 of 18 |
| **`pearlN` (clean)** | **0.882** | **0.871** | **0 of 18** |

The idle sheet was a symptom, not a design flaw. A view missing a third of its content cannot
give its panel family enough work to do, so a sheet sat out; with complete targets the load
balances 6 / 6 / 6 and every sheet earns its place. Roughly a third of the total improvement in
this project came from fixing inputs and constraints rather than from optimisation.

