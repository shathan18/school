# New-style pair render results (300-shard, lab-gated)

Recipe: PC=14, seed=2, dw=0.5, cw=1.0, match_metric='lab', solver_tol=20.0, report_tol=25.0.

| # | label | SSIM A/B | edge A/B | good A/B | bad A/B | good/bad | shards | elapsed |
|---|-------|:--------:|:--------:|:--------:|:-------:|:--------:|:------:|:-------:|
| 1 | mondrian_ii__mondrian_iii | 0.745/0.818 | 0.453/0.552 | 0.0/0.3% | 0.0/5.4% | 0.06 | 209 | 4.3s |
| 2 | dawn_isawa__mondrian_iii | 0.641/0.830 | 0.239/0.596 | 0.4/0.6% | 2.4/7.7% | 0.10 | 204 | 4.6s |
| 3 | dawn_isawa__goten_yama | 0.602/0.565 | 0.168/0.309 | 14.9/8.1% | 13.7/19.4% | 0.70 | 290 | 5.1s |
| 4 | fuji_a__wpa_yellowstone | 0.721/0.583 | 0.417/0.272 | 4.1/0.9% | 13.4/4.7% | 0.27 | 247 | 4.1s |