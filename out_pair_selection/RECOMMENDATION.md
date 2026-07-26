# Pair selection: blur pre-test → palette pairing → 300-shard renders

**Criterion (per lecturer):** an image survives the 300-flat-shard regime only if it
reduces to **one dominant bold shape on a contrasting field with few colours**. Red
Fuji survived (red triangle / blue sky). Great Wave died (mid-scale detail, no
silhouette). We now select on this measured property, not fame.

**Constraints applied:** public domain, secular, no depictions of women, no
political/religious/mythological content. Buddhist temple views excluded from the
Hokusai series pool (`asakusa_honganji`, `sazai_hall`).

## Pipeline

1. `blur_pretest.py` — heavy Gaussian blur (σ=12 on 256px) + K=4 posterize; scores
   each candidate on `{subj_compact, n_big_regions, top2_frac, contrast_dE, subj_area}`
   and gates on `S >= 0.35, subj_compact >= 0.20, n_big_regions <= 7,
subj_area in [0.08, 0.75]`. Verified: Red Fuji PASSES (S=0.42), Great Wave FAILS
   (S=0.02), fuji_a PASSES (S=1.06, best), busy townscapes (Yoshida, Senju, Sumida)
   FAIL. → `out_pair_selection/pretest_contact.png`, `pretest_scores.jsonl`,
   `pretest_summary.md`.
2. `pair_score.py` — EMD-in-Lab between blur-survivors' posterized palettes, with
   shared-series (+15%) and dominant-cluster-agreement (+15%) bonuses. Combined score =
   palette-compat × geometric mean of the two blur scores. → `pair_scores.tsv`,
   `pair_palettes.png`.
3. `render_pair.py` — full 300-shard render (PC=14, diag 30–60°, dw=0.5, cw=1.0,
   match_metric='lab', solver_tol=20). Reports `colour_agreeing_duty` (good) and
   wrong-colour bleed (bad) at report_tol=25, per wall. → `pairs/<label>/preview_final.png`,
   `scene_interactive.html`, `metrics.json`, `pair_render_summary.md`.

## Phase 1: blur pre-test — 9 survivors of 52

Ranked top survivors (see `pretest_summary.md` for the full table):

| #   | name                                             | score | subj_compact | n_big_regions | contrast_dE |
| --- | ------------------------------------------------ | ----: | -----------: | ------------: | ----------: |
| 1   | dawn_at_isawa_in_the_kai_province                | 1.085 |         1.00 |             4 |          58 |
| 2   | fuji_a                                           | 1.060 |         1.00 |             5 |          58 |
| 3   | kajikazawa                                       | 0.754 |         0.79 |             5 |          49 |
| 4   | at_sea_off_kazusa                                | 0.589 |         1.00 |             6 |          41 |
| 5   | goten_yama_hill_shinagawa                        | 0.570 |         0.98 |             6 |          32 |
| 6   | tempesta_sotto_la_vetta (storm below the summit) | 0.461 |         0.43 |             4 |          67 |
| 7   | red_fuji / red_fuji_southern_wind_clear_morning  | 0.417 |         0.70 |             4 |          28 |
| 8   | fuji_c                                           | 0.412 |         0.46 |             5 |          42 |

**Rejected famous ones (blunt):** Great Wave off Kanagawa (S=0.020 — too much
mid-scale foam, no single silhouette), h_amida_falls (S=0.065 — the waterfall bar is
thin), h_kirifuri_falls (S=0.033 — figures clutter the falls), enoshima standalone
(S=0.020 — flat headland, no shape), most Hiroshige-style series townscapes (Yoshida,
Senju, Umegawa, Ushibori — all S<0.02).

## Phase 2: palette-compatible pairs — top-3 to render

From `pair_scores.tsv` (top rows):

| #   | pair                                                            | combined | palette_EMD | dominant_dE |
| --- | --------------------------------------------------------------- | -------: | ----------: | ----------: |
| 1   | dawn_at_isawa × goten_yama_hill_shinagawa (both Hokusai series) |    0.839 |        19.4 |        13.6 |
| 2   | dawn_at_isawa × fuji_a (mixed source)                           |    0.795 |        25.9 |        48.3 |
| 3   | dawn_at_isawa × kajikazawa (both bold Fuji-region silhouettes)  |    0.758 |        16.1 |        18.0 |

Note `dawn_at_isawa` wins Phase 1 and anchors all top-3 pairs — its bold blue-mountain-on-cream
palette is a compatibility magnet inside this pool.

## Phase 3: 300-shard render results — RANKED RECOMMENDATION

Reference (from `report_team.md` + `out_noise_study/`): palette-compatible pairs land
good/bad ≈ 0.88–0.93; arbitrary pairs ≈ 0.27–0.35.

| #    | pair                                      | SSIM A/B  | edge A/B  |  good A/B  |  bad A/B   | good/bad | shards |
| ---- | ----------------------------------------- | :-------: | :-------: | :--------: | :--------: | :------: | :----: |
| 🥇 1 | dawn_at_isawa × kajikazawa                | 0.62/0.59 | 0.20/0.24 | 12.0/11.1% | 13.7/16.7% | **0.76** |  291   |
| 🥈 2 | dawn_at_isawa × fuji_a                    | 0.61/0.74 | 0.21/0.52 | 9.7/12.2%  | 11.9/18.9% | **0.71** |  265   |
| 🥉 3 | dawn_at_isawa × goten_yama_hill_shinagawa | 0.60/0.57 | 0.17/0.31 | 14.9/8.1%  | 13.7/19.4% | **0.70** |  290   |

Interactive previews: `pairs/<label>/scene_interactive.html`.

### Blunt read

- **All three pairs sit in a 0.70–0.76 good/bad band** — clearly above the "arbitrary"
  ratio (~0.30) but _not_ at the palette-fully-compatible "gold" ratio (~0.90). The
  small survivor pool (9 of 52) doesn't contain a truly matched-palette pair the way
  Girl-front/back or Pearl-front/back do. Honest ceiling for this pool at 300 shards
  is "readable + moderately colour-compatible", not "spectacular".
- **Pick #1 (dawn_at_isawa × kajikazawa)** as the best delivery candidate: highest
  good/bad ratio, best palette EMD (16.1 — tightest match in the pool), both images
  are single-mountain silhouettes so _reading_ on the walls should feel like a
  matched pair.
- **Pick #2 (dawn_at_isawa × fuji_a)** if raw fidelity matters more: it wins the
  edge-fidelity metric decisively (edge_B=0.517 vs 0.24/0.31 for the others) because
  fuji_a is a hand-simplified poster cut and pushes very cleanly.
- **Pick #3** is a same-Hokusai-series pair by construction (both from _36 Views_)
  but its good/bad ratio is lowest — same-series ≠ same-palette in practice; Hokusai's
  colouring across the series varies more than expected.

### What we deliberately did NOT do (per criterion)

- Full-render every palette pair — that would burn compute on losers already killed
  by the blur pre-test. The whole design of Phase 1 is to cut waste.
- Include famous-but-rejected pieces (Great Wave, Kirifuri Falls, most Hiroshige
  townscapes). Fame is not the criterion; survivability is.

## Reproduce

```powershell
py blur_pretest.py         # 52 candidates -> 9 survivors + contact sheet
py pair_score.py           # 28 pair-scores + palette side-by-side
py render_pair.py          # top-3 pairs, 300-shard render each, metrics.json
```

Outputs land in `out_pair_selection/`.
