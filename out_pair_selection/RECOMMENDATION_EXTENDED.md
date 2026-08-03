# Pair selection, extended: new styles beyond the Hokusai pool

Extends the team's blur-survivability pipeline (`blur_pretest.py` → `pair_score.py` →
`render_pair.py`, unmodified) with 10 new candidates spanning Hiroshige, Mondrian, WPA
Art Deco posters, and Caspar David Friedrich — testing the "single dominant bold shape,
survives blur" criterion outside the original Japanese-woodblock-only pool.

**Sourcing (public domain, verified per-file on Wikimedia Commons):** all 10 downloaded
from confirmed Commons file pages. Hiroshige (d. 1858) and Friedrich (d. 1840) are
unambiguously PD. Mondrian works chosen are the ones Commons tags PD in the US (published
before 1931). WPA National Park posters are PD as works of the US federal government
(Works Progress Administration), regardless of age — the cleanest PD case of the batch.
**Matisse cutouts were deliberately excluded**: most are not yet public domain in the US
(published 1940s, 95-year term), so including them would have been a real copyright risk,
not just a stylistic gap. Botanical silhouettes were also skipped — no specific single-work
candidate could be verified with confidence in the time available, and guessing a filename
was not an acceptable substitute for verification.

## Phase 1: extended blur pre-test — 62 candidates (52 original + 10 new), 14 survivors

Full contact sheet: `pretest_contact.png`. Full table: `pretest_summary.md`
(regenerated in place; scores for the original 52 are unchanged, since the criterion,
knees, and scoring code are byte-identical to the team's original `blur_pretest.py` —
only the candidate pool was extended, via `blur_pretest_extended.py`).

**New candidates, ranked:**

| name | score | pass/fail | note |
|---|---:|---|---|
| mondrian_comp_iii_1929 | 0.962 | **PASS** | 2nd-highest score of all 62 candidates |
| wpa_yellowstone_1938 | 0.675 | **PASS** | Old Faithful geyser plume — bold white-on-blue |
| mondrian_comp_ii_red_blue_yellow_1930 | 0.631 | **PASS** | |
| mondrian_comp_red_yellow_blue_black_1921 | 0.448 | **PASS** | |
| wpa_lassen_volcanic_1938 | 0.442 | **PASS** | |
| hiroshige_shin_ohashi_shower | 0.321 | fail | close, but rain-streak striping adds too many small regions |
| wpa_grand_canyon_1938 | 0.081 | fail | layered canyon strata don't reduce to one shape |
| hiroshige_plum_garden_kameido | 0.105 | fail | the famous branch silhouette still fragments into many small regions |
| friedrich_monk_by_the_sea | 0.057 | fail | surprising — see below |
| hiroshige_fukagawa_susaki_hawk | 0.008 | fail | |

**Blunt read on the surprises:** two "obvious" bold candidates I expected to pass, failed.
*Monk by the Sea* — probably the most minimalist composition in Western painting — fails
because the actual reproduction's sky/sea bands sit too close in Lab lightness/hue for the
contrast-dE gate, and the monk figure is too small/thin to register as a qualifying
silhouette region. *Plum Garden, Kameido* fails because the branch's fine forking structure
reads as many small connected components, not one bold shape, under this metric — visually
iconic is not the same as "reduces to one region" under strong blur. Cut both, per the
brief's own instruction to prioritize survivability over fame.

**Mondrian and two of three WPA posters pass outright** — unsurprising in hindsight
(geometric abstraction and flat-colour propaganda-poster design are close to *already
being* posterized), but not previously tested in this project.

## Phase 2: palette-compatible pairs among all 13 de-duplicated survivors

Full ranked table: `pair_scores.tsv` (78 pairs). Top of the list is still dominated by the
existing Hokusai survivors (`dawn_at_isawa`, `fuji_a`, `kajikazawa`, `goten_yama`), but new
cross-movement pairs surface from rank 7 onward, including a Hokusai×Mondrian pairing and
several Mondrian/WPA combinations.

## Phase 3: full 300-shard renders — 4 pairs

Rendered via the team's unmodified `render_pair.py` (`scenes/example.yaml`, PC=14,
match_metric='lab', solver_tol=20, report_tol=25) — same recipe as the original three:

| # | pair | SSIM A/B | edge A/B | good A/B | bad A/B | good/bad | shards |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | dawn_at_isawa × goten_yama (reference) | 0.60/0.57 | 0.17/0.31 | 14.9/8.1% | 13.7/19.4% | 0.70 | 290 |
| 2 | fuji_a × wpa_yellowstone (NEW) | 0.72/0.58 | 0.42/0.27 | 4.1/0.9% | 13.4/4.7% | 0.27 | 247 |
| 3 | dawn_at_isawa × mondrian_iii (NEW) | 0.64/0.83 | 0.24/0.60 | 0.4/0.6% | 2.4/7.7% | 0.10 | 204 |
| 4 | mondrian_ii × mondrian_iii (NEW) | 0.75/0.82 | 0.45/0.55 | 0.0/0.3% | 0.0/5.4% | 0.06 | 209 |

### The good/bad ratio is misleading for this whole class of pair — verified by eye and by ceiling/straddle

Ranked by good/bad ratio alone, the three new pairs look like the *worst* results —
below even the "arbitrary pair" reference band (~0.27–0.35). **Looking at the actual
`preview_final.png` renders overturns this completely**: pairs #2 and #4 are the two
cleanest, most legible reconstructions produced anywhere in this project. Pair #1 (the
metric's "winner") is visibly the muddiest of the four — both walls fragment into
indistinct mottled colour on inspection.

`ceiling_straddle.py` (team's script, paths/scene corrected, unmodified logic) explains
why:

| pair | wall | B_good | ceil(a) | ceil(c) | straddle |
|---|---|---:|---:|---:|---:|
| dawn_isawa × goten_yama | A | 17.3% | 22.1% | 32.2% | 78% |
| | B | 6.1% | 7.0% | 25.4% | 75% |
| fuji_a × wpa_yellowstone | A | 6.7% | 5.1% | 27.5% | **89%** |
| | B | 5.3% | 6.8% | 16.5% | 78% |
| mondrian_ii × mondrian_iii | A | 1.7% | 2.9% | 17.6% | **95%** |
| | B | 0.0% | 0.3% | 1.0% | 68% |

**straddle near 90–95% means shape/geometry is essentially not the bottleneck for the new
pairs** — landing blobs are colour-uniform almost everywhere. Their low "good" score isn't
messy placement; it's that these bold images have so *few* distinct colours that a stray
shadow's colour rarely coincides with the small number of specific hues the other wall
wants at that exact point. But because each region is large and flat, even "wrong-colour"
bleed lands as **one clean, uniform patch**, not chaotic noise — which is exactly what the
eye sees: a crisp, if occasionally off-hue, picture, never mud.

**Conclusion: the "good/bad ratio ≈0.9 compatible / ≈0.3 arbitrary" heuristic was
calibrated on continuous-tone photographic/painterly pairs (Mona Lisa, Monet), where
cross-talk exploitation is doing real work. It does not transfer to bold-graphic/poster
pairs, where the real success factor is low absolute contamination from having few, large,
flat regions — not a high exploited-cross-talk ratio.** For this class of pair, rank by
**bad% (lower better) and straddle (higher = geometry is not capping you)**, not by the
good/bad ratio.

## Ranked recommendation

1. 🥇 **fuji_a × wpa_yellowstone (NEW)** — the standout. Both walls read almost
   instantly (a distinct snow-capped blue mountain; an unmistakable geyser plume). Lowest
   `bad` on wall B (4.7%) of any pair tested. A genuine cross-cultural echo: a Japanese
   ukiyo-e mountain cut and a 1930s American federal travel poster, from opposite sides of
   the world, turn out to be graphically the same idea.
2. 🥈 **mondrian_ii × mondrian_iii (NEW)** — same-artist pair, and it shows: both walls
   reconstruct their red/blue/yellow blocks almost exactly. Lowest absolute contamination
   of everything tested (bad_B 5.4%, bad_A ~0%). The purest proof of the "bold shape, few
   colours" criterion — this is close to the ceiling of what the medium can do.
3. 🥉 **dawn_at_isawa × goten_yama (reference, unchanged)** — kept for continuity with the
   team's original pick, but the eye-check says it is now the weakest of the four
   rendered; recommend it drop to third if a hard cut to two is wanted.

**Reproduce:**
```
py blur_pretest_extended.py   # 62 candidates -> 14 survivors + contact sheet
py pair_score.py              # unchanged, re-run on extended survivor pool
py render_new_pairs.py        # 4 pairs, 300-shard render each
py ceiling_straddle_new.py    # ceiling/straddle diagnostic on 3 of the 4
```
