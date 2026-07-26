# Pair render results (300-shard, lab-gated)

Recipe: PC=14, seed=2, dw=0.5, cw=1.0, match_metric='lab', solver_tol=20.0, report_tol=25.0.

Ranked by `good_mean - bad_mean` (net colour-agreeing double duty).

| # | label | SSIM A/B | edge A/B | good A/B | bad A/B | good/bad | shards | elapsed |
|---|-------|:--------:|:--------:|:--------:|:-------:|:--------:|:------:|:-------:|
| 1 | dawn_at_isawa_in_the_kai_p__kajikazawa | 0.621/0.587 | 0.197/0.237 | 12.0/11.1% | 13.7/16.7% | 0.76 | 291 | 3.5s |
| 2 | dawn_at_isawa_in_the_kai_p__fuji_a | 0.607/0.735 | 0.206/0.517 | 9.7/12.2% | 11.9/18.9% | 0.71 | 265 | 3.4s |
| 3 | dawn_at_isawa_in_the_kai_p__goten_yama_hill_shinagawa_ | 0.602/0.565 | 0.168/0.309 | 14.9/8.1% | 13.7/19.4% | 0.70 | 290 | 4.0s |

## Interpretation

- **good/bad ratio** is the palette-compat surrogate. Reference (from `report_team.md` /
  `out_noise_study/`): palette-compatible pairs land ~0.88..0.93, arbitrary pairs ~0.27..0.35.
- **SSIM/edge** are per-wall fidelity. Absolute SSIM ~=0.68..0.72 is normal at 300 shards.
- **joint@0.2** is colour-BLIND (reads ~100% for every layout) - context only, not a ranker.