# Lowering cross-talk noise without losing double duty

**Question.** The sculpture is meant to have shards that genuinely serve *both* wall images.
That inevitably produces stray shadow on the other wall. How do we keep the double duty while
cutting the noise it drags along?

**Answer in one line.** The credit gate in `decompose._shard_damage` judges "is this the right
colour?" in **raw RGB**, which is blind to hue at low luminance, so it pays double-duty credit
to shards that actually read as coloured stains on dark regions. Switching that gate to
**CIELAB ΔE** cuts wrong-colour bleed by 6–28% while holding genuine duty — and *improves*
SSIM and edge-fidelity at the same time.

Evidence: 174 runs across 3 studies, 2–3 image pairs, 3 seeds each, fixed 14-panel diagonal
geometry (30–60°). Raw data in `out_noise_study/`.

---

## 1. The metric we had been reading cannot see noise

The panel-count sweep ranked layouts partly on
`panel_search.joint_intersection_pct`. That metric is **colour-blind**: it counts any stray
shadow that *lands on* the other image's subject, whatever colour it arrives in.

It reported **100.0%** for all 174 configurations tested here — including deliberately bad
ones (`damage_weight=0.0`, i.e. random shard assignment). It cannot discriminate, so it must
not be used to rank anything.

This is the same failure already recorded in `corrections_note.md` §3 (the 27–31% → ~3%
correction). The honest measure is `search.colour_agreeing_duty`, which renders **only the
panels whose primary wall is the other one** — pure cross-talk — and splits the subject pixels
it darkens:

| term | meaning |
| --- | --- |
| **good** | stray shadow arrived in ~the colour that wall wants → genuine double duty |
| **bad**  | stray shadow arrived in the wrong colour → **this is the noise** |

Both are expressed as % of that wall's subject area.

> **Reporting discipline.** Every arm below is judged at one *fixed* ruler (CIELAB ΔE < 25),
> even when the arm's own internal optimiser gate differs. Otherwise an arm can "win" simply by
> grading itself more leniently.

---

## 2. Why the noise is being actively rewarded

`_shard_damage` (`shadowart/solve/decompose.py`, ~line 598) scores a candidate placement as a
**signed** quantity:

```python
e_without = ((1.0 - tgt) ** 2).sum(axis=1)      # cost of leaving the pixel blank white
e_with    = ((transmit[None, :] - tgt) ** 2).sum(axis=1)
signed    = e_with - e_without                  # negative == credit == "this shard helps"
```

On a **dark** subject pixel, `e_without` is large — leaving it white is very wrong — so almost
*any* dark shard scores negative and earns credit, **regardless of its hue**.

The `match_tol` gate exists to stop exactly that. But it defaults to `match_metric="rgb"`, and
raw-RGB Euclidean distance collapses hue differences at low luminance:

| comparison | RGB distance | CIELAB ΔE |
| --- | --- | --- |
| dark neutral target `(0.10, 0.10, 0.10)` vs dark red shard `(0.30, 0.05, 0.05)` | **0.21** → passes the 0.30 gate | **≈ 35** → blocked by a ΔE 15 gate |

So the shipped gate hands "double duty" credit to shards that a viewer reads as **red stains on
a dark region**. That is the noise, and the optimiser is paying for it.

CIELAB is roughly perceptually uniform, so ΔE tracks what the eye sees and separates hue at low
luminance. `targets/color.py` already ships `rgb_to_lab` and `delta_e`, and
`fragment_shards_overlap` already accepts `match_metric="lab"` — no new algorithm code is
required.

---

## 3. The fix, and where its knee is

`noise_study_gate.py` — 14 panels, angles 30–60°, seeds 1/2/3, `damage_weight=0.5`,
`credit_weight=1.0`. Δ columns are relative to the current `rgb 0.30` default.

### wave_fuji (arbitrary pair)

| gate | good % | bad % | good/bad | Δbad | Δgood | SSIM | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rgb 0.30 *(current)* | 5.00 | 18.30 | 0.27 | — | — | 0.683 | 0.477 |
| lab ΔE30 | 5.82 | 17.82 | 0.33 | −2.6% | +16.4% | 0.682 | 0.469 |
| lab ΔE20 | 4.92 | 15.23 | 0.32 | **−16.8%** | −1.7% | 0.689 | 0.489 |
| lab ΔE15 | 4.80 | 14.65 | 0.33 | **−19.9%** | −4.0% | 0.690 | 0.487 |
| **lab ΔE12** | 4.73 | 13.17 | 0.36 | **−28.0%** | −5.4% | **0.690** | 0.486 |
| lab ΔE9 | 4.35 | 11.35 | 0.38 | −38.0% | −13.1% | 0.692 | 0.490 |
| lab ΔE6 | 4.14 | 10.67 | 0.39 | −41.7% | −17.1% | 0.693 | 0.494 |

### pearl_earring (palette-compatible pair)

| gate | good % | bad % | good/bad | Δbad | Δgood | SSIM | edge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rgb 0.30 *(current)* | 17.74 | 20.19 | 0.88 | — | — | 0.745 | 0.501 |
| lab ΔE30 | 18.20 | 21.64 | 0.84 | +7.2% | +2.6% | 0.745 | 0.496 |
| **lab ΔE20** | 17.77 | 19.02 | **0.93** | **−5.8%** | **+0.2%** | 0.746 | **0.512** |
| lab ΔE15 | 16.46 | 18.65 | 0.88 | −7.7% | −7.2% | 0.748 | 0.530 |
| lab ΔE12 | 14.84 | 18.79 | 0.79 | −6.9% | −16.3% | 0.751 | 0.547 |
| lab ΔE9 | 13.39 | 18.59 | 0.72 | −7.9% | −24.5% | 0.753 | 0.553 |
| lab ΔE6 | 11.23 | 18.29 | 0.61 | −9.4% | −36.7% | 0.756 | 0.567 |

### Reading the knee

- **SSIM and edge-fidelity rise monotonically as the gate tightens.** This is *not* a tradeoff
  against image quality. Blocking fake credit also stops the solver from parking wrong-colour
  shards, which sharpens the primary wall.
- **Arbitrary pairs tolerate a tight gate.** There is almost no genuine agreement to protect, so
  tightening strips fake credit almost exclusively: ΔE12 buys −28% noise for −5% duty.
- **Compatible pairs need a looser gate.** Real agreement exists, so below ΔE≈20 the gate starts
  blocking *genuine* matches — duty falls faster than noise (ΔE12: −16.3% duty for −6.9% noise).

**Recommended setting**

```python
match_metric = "lab"
match_tol    = 20.0   # palette-compatible pairs (same subject / same palette)
match_tol    = 12.0   # arbitrary pairs
```

- Raising `credit_weight` to 2.0 to buy back the duty a tight gate costs does **not** help
  materially (pearl ΔE12 c2.0: duty −11.7% vs −16.3%, but noise recovers from −6.9% to −5.4%).
  It moves both halves together rather than separating them.

---

## 4. Negative results (both worth keeping)

### 4.1 Outline protection — no effect

`decompose.outline_map` + `outline_protect_weight` is documented as the fix for `_shard_damage`
being edge-blind: a stray shadow on a dark contour steals almost no light, so it scores harmless
even though it is destroying the line that carries the image.

Measured effect: **none.**

| arm | pearl bad % | wave bad % |
| --- | ---: | ---: |
| baseline | 20.19 | 18.30 |
| `outline_protect_weight=1.0` | 20.22 | 18.25 |
| `outline_protect_weight=3.0` | 20.23 | 18.25 |

Combined with the LAB gate it also adds nothing beyond what the gate already delivers
(`lab_dE15+outline3` ≈ `lab_dE15`). The docstring oversells this lever.

### 4.2 `spill_weight` at panel placement — backfires

`panel_search._coverage_score` carries an unused subject-aware term that makes the greedy prefer
panels whose footprint lands on *subject* on both walls:

```python
spill_frac = spill / (spill + gain)
score      = gain * (1.0 - spill_frac) ** spill_weight    # spill_weight=0.0 == spill-blind
```

The hypothesis was that noise is decided upstream — if every panel splatters onto the other
wall's background, no assignment rule can rescue it. The data says otherwise.

| pair | spill_weight | good % | bad % | SSIM | edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| pearl | 0.0 | 17.74 | 20.19 | 0.745 | 0.501 |
| pearl | 1.0 | 17.48 | 22.08 | 0.738 | 0.505 |
| pearl | 2.0 | 17.87 | 23.12 | 0.728 | 0.449 |
| pearl | 4.0 | 17.24 | **23.56** | **0.719** | **0.418** |
| wave | 0.0 | 5.00 | 18.30 | 0.683 | 0.477 |
| wave | 4.0 | 7.24 | **22.58** | 0.680 | 0.475 |

It raises **good and bad together** and costs real fidelity. Pointing more shadow at the other
wall's subject does not make that shadow the right *colour*.

> **Conclusion: geometry is not the bottleneck — colour is.** This is why the assignment-time
> colour gate is the lever that works and the placement-time geometric lever is not.

### 4.3 `harm_only` — the known trap, re-confirmed

Damage-only assignment (`credit_weight=None`) does crush the noise, but by refusing all double
duty and collapsing onto a fraction of the panels:

| arm | pearl good % | pearl bad % | panels used (of 14) |
| --- | ---: | ---: | ---: |
| random (`damage_weight=0.0`) | 6.43 | 13.11 | 14.0 |
| harm_only | 2.01 | **5.01** | **6.7** |
| baseline signed credit | 17.74 | 20.19 | 12.0 |

Low noise here is not a win — it is the sculpture refusing to do the thing it exists to do.

---

## 5. The dominant factor is the image pair, not the tuning

Across every arm tested, pair palette-compatibility swamps all solver settings:

| pair | good % (range) | good/bad (range) |
| --- | --- | --- |
| pearl front/back — same figure, same palette | 16–18% | 0.88–0.93 |
| wave / blue fuji — arbitrary | 4–7% | 0.27–0.39 |
| apples / breakfast — arbitrary | ~0% | ~0 |

`apples_breakfast` yields **0.0% genuine duty on both walls at every setting tested.** The two
images do not want the same colours at geometrically-linked points, so no gate, weight, or
placement rule can manufacture agreement that is not in the source material.

This restates `corrections_note.md` §4 and is the single most actionable finding: **choose
palette-compatible sources up front.** That choice buys more real double duty than any optimiser
change measured here.

---

## 6. What was changed

`render_best_panels.py`:

```python
MATCH_METRIC      = "lab"
MATCH_TOL_DEFAULT = 12.0                 # arbitrary pairs
MATCH_TOL         = {"pearl_earring": 20.0}   # palette-compatible pairs
```

It now also reports the honest good/bad split alongside the (retained but uninformative)
colour-blind joint figure.

Re-rendered previews in `out_panel_sweep/`:

| pair | config | duty A/B | noise A/B | SSIM A/B | previous SSIM A/B |
| --- | --- | --- | --- | --- | --- |
| pearl_earring | pc=6 seed=2 | 28.6% / 5.4% | 12.4% / **34.9%** | 0.746 / 0.781 | 0.748 / 0.780 |
| wave_fuji | pc=22 seed=1 | 2.9% / 7.8% | 7.7% / 10.6% | **0.751 / 0.647** | 0.744 / 0.640 |
| apples_breakfast | pc=22 seed=2 | 0.0% / 0.0% | 16.7% / 21.2% | **0.833 / 0.853** | 0.825 / 0.844 |

---

## 7. Open items

1. **Per-wall tolerance.** pearl's noise is strongly asymmetric — 12.4% on Wall A vs **34.9%** on
   Wall B. A single global `match_tol` cannot serve both; `match_tol` should probably be a
   per-wall value tuned to each target's luminance distribution.
2. **Replace the ranking metric.** `sweep_panels.py` still scores with the colour-blind
   `joint_intersection_pct` term. It should use `colour_agreeing_duty` (`+w·good − w·bleed`), which
   is what `search.score_layout` already does when given `duty`/`bleed`.
3. **Why outline protection is inert.** Worth one debugging pass — either the protect map is
   effectively zero where it matters, or `protect_weight` is being swamped by the signed term.

---

## 8. Reproducing

```powershell
cd F:\Studies\school
py noise_study.py         # lever factorial: gates, outline protection, credit weight
py noise_study_spill.py   # placement-time spill_weight x assignment gate
py noise_study_gate.py    # ΔE tolerance knee
py render_best_panels.py  # re-render winning previews with the LAB gate
```

Outputs: `out_noise_study/{runs,spill_runs,gate_runs}.jsonl` plus matching `.log` files.

**Fixed conditions.** `scenes/example.yaml`, 14 panels, `angle_deg_range=(30, 60)`,
`K_CANDIDATES=16`, `damage_weight=0.5`, seeds 1/2/3. Panel geometry is built **once per seed,
before** the arm loop, so the only thing varying between arms is shard→panel assignment.
