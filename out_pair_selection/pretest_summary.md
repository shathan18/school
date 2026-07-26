# Blur pre-test results

Pool: 52 candidates.  Passed: **9**.  Failed: 43.

Config: downsample to 256px long edge, Gaussian sigma=12.0, K-means K=4 in CIELAB.
Pass gates: score>=0.35, subj_compact>=0.2, n_big_regions<=7, subj_area in [0.08, 0.75].

## Survivors (ranked)

| # | name | score | subj_compact | n_big_regions | top2_frac | contrast dE | subj_area | palette used |
|---|------|------:|-------------:|--------------:|----------:|------------:|----------:|-------------:|
| 1 | dawn_at_isawa_in_the_kai_province | 1.085 | 1.00 | 4 | 0.58 | 58.5 | 0.30 | 4 |
| 2 | fuji_a | 1.060 | 1.00 | 5 | 0.71 | 57.5 | 0.34 | 4 |
| 3 | kajikazawa | 0.754 | 0.79 | 5 | 0.65 | 49.0 | 0.36 | 4 |
| 4 | at_sea_off_kazusa_kazusa_no_kairo_from_the_serie | 0.589 | 1.00 | 6 | 0.55 | 41.4 | 0.19 | 4 |
| 5 | katsushika_hokusai_goten_yama_hill_shinagawa_on_ | 0.570 | 0.98 | 6 | 0.67 | 32.4 | 0.37 | 4 |
| 6 | katsushika_hokusai_tempesta_sotto_la_vetta_dalla | 0.461 | 0.43 | 4 | 0.59 | 67.2 | 0.36 | 4 |
| 7 | red_fuji_southern_wind_clear_morning | 0.417 | 0.70 | 4 | 0.52 | 27.7 | 0.27 | 4 |
| 8 | red_fuji | 0.417 | 0.70 | 4 | 0.52 | 27.7 | 0.27 | 4 |
| 9 | fuji_c | 0.412 | 0.46 | 5 | 0.66 | 41.8 | 0.35 | 4 |

## Rejected (ranked by score, best-first)

| name | score | subj_compact | n_big_regions | top2_frac | contrast dE | subj_area | reason |
|------|------:|-------------:|--------------:|----------:|------------:|----------:|--------|
| enoshima_in_the_sagami_province | 0.216 | 0.55 | 6 | 0.67 | 21.9 | 0.39 | S<0.35 |
| fuji_b | 0.196 | 0.17 | 4 | 0.70 | 44.4 | 0.18 | S<0.35 cmp<0.2 |
| sunset_across_the_ryogoku_bridge_from_the_bank_o | 0.171 | 0.30 | 5 | 0.63 | 29.7 | 0.19 | S<0.35 |
| shichirigahama_in_sagami_province_ssh_shichiriga | 0.167 | 0.57 | 5 | 0.56 | 15.5 | 0.29 | S<0.35 |
| nakahara_in_the_sagami_province | 0.151 | 0.34 | 6 | 0.53 | 30.1 | 0.28 | S<0.35 |
| soshu_nakahara | 0.147 | 0.44 | 6 | 0.61 | 20.7 | 0.39 | S<0.35 |
| reflection_in_lake_at_misaka_in_kai_province_ksh | 0.145 | 0.52 | 6 | 0.57 | 17.9 | 0.35 | S<0.35 |
| ejiri_in_suruga_province_sunsh_ejiri_from_the_se | 0.129 | 0.30 | 5 | 0.52 | 26.2 | 0.22 | S<0.35 |
| under_the_mannen_bridge_at_fukagawa_fukagawa_man | 0.114 | 0.29 | 6 | 0.53 | 26.4 | 0.32 | S<0.35 |
| the_tea_plantation_of_katakura_in_the_suruga_pro | 0.103 | 0.29 | 4 | 0.67 | 14.7 | 0.45 | S<0.35 |
| noboto_bay_noboto_no_ura_from_the_series_thirty_ | 0.095 | 0.57 | 6 | 0.58 | 10.5 | 0.23 | S<0.35 |
| the_back_of_the_fuji_from_the_minobu_river | 0.085 | 0.25 | 6 | 0.56 | 21.3 | 0.30 | S<0.35 |
| nihonbashi_in_edo_edo_nihonbashi_from_the_series | 0.070 | 0.30 | 7 | 0.51 | 19.4 | 0.29 | S<0.35 |
| h_amida_falls | 0.065 | 0.08 | 4 | 0.59 | 35.2 | 0.28 | S<0.35 cmp<0.2 |
| climbing_on_mt_fuji | 0.056 | 0.27 | 9 | 0.48 | 24.1 | 0.24 | S<0.35 nreg>7 |
| the_fuji_from_kanaya_on_the_tokaido | 0.049 | 0.17 | 6 | 0.56 | 18.1 | 0.32 | S<0.35 cmp<0.2 |
| kajikazawa_in_kai_province_koshu_kajikazawa | 0.045 | 0.14 | 7 | 0.65 | 21.4 | 0.24 | S<0.35 cmp<0.2 |
| a_sketch_of_the_mitsui_shop_in_suruga_street_in_ed | 0.037 | 0.15 | 9 | 0.50 | 28.5 | 0.22 | S<0.35 cmp<0.2 nreg>7 |
| surugadai_in_edo_tto_sundai_from_the_series_thir | 0.036 | 0.14 | 6 | 0.54 | 17.8 | 0.34 | S<0.35 cmp<0.2 |
| sekiya_village_on_the_sumida_river_sumidagawa_se | 0.034 | 0.28 | 8 | 0.49 | 12.4 | 0.21 | S<0.35 nreg>7 |
| h_kirifuri_falls | 0.033 | 0.24 | 9 | 0.49 | 15.0 | 0.28 | S<0.35 nreg>7 |
| a_sketch_of_the_mitsui_shop_in_suruga_street_in_ | 0.027 | 0.07 | 7 | 0.60 | 26.9 | 0.31 | S<0.35 cmp<0.2 |
| senju_in_musashi_province_bush_senju_from_the_se | 0.023 | 0.13 | 8 | 0.56 | 15.6 | 0.34 | S<0.35 cmp<0.2 nreg>7 |
| the_lake_of_hakone_in_the_segami_province | 0.022 | 0.17 | 8 | 0.32 | 21.6 | 0.17 | S<0.35 cmp<0.2 nreg>7 |
| great_wave_off_kanagawa2 | 0.020 | 0.08 | 6 | 0.64 | 13.5 | 0.23 | S<0.35 cmp<0.2 |
| enoshima | 0.020 | 0.09 | 5 | 0.61 | 11.4 | 0.35 | S<0.35 cmp<0.2 |
| honjo_tatekawa_the_timberyard_at_honjo | 0.019 | 0.09 | 8 | 0.49 | 21.6 | 0.26 | S<0.35 cmp<0.2 nreg>7 |
| the_coast_of_seven_leages_in_kamakura | 0.017 | 0.09 | 11 | 0.42 | 31.6 | 0.20 | S<0.35 cmp<0.2 nreg>7 |
| lower_meguro_shimo_meguro_from_the_series_thirty | 0.017 | 0.08 | 5 | 0.66 | 9.5 | 0.37 | S<0.35 cmp<0.2 |
| in_the_mountains_of_ttomi_province_ttomi_sanch_f | 0.017 | 0.07 | 5 | 0.68 | 12.2 | 0.46 | S<0.35 cmp<0.2 |
| tago_bay_near_ejiri_on_the_tkaid_tkaid_ejiri_tag | 0.016 | 0.22 | 8 | 0.34 | 10.8 | 0.17 | S<0.35 nreg>7 |
| yoshida_on_the_tkaid_tkaid_yoshida_from_the_seri | 0.015 | 0.09 | 6 | 0.60 | 10.6 | 0.40 | S<0.35 cmp<0.2 |
| the_waterwheel_at_onden_onden_no_suisha_from_the | 0.014 | 0.12 | 7 | 0.52 | 10.0 | 0.35 | S<0.35 cmp<0.2 |
| fujimigahara_in_owari_province_bish_fujimigahara | 0.012 | 0.10 | 8 | 0.60 | 9.9 | 0.37 | S<0.35 cmp<0.2 nreg>7 |
| ushibori_in_hitachi_province_jsh_ushibori_from_t | 0.010 | 0.05 | 8 | 0.63 | 15.5 | 0.38 | S<0.35 cmp<0.2 nreg>7 |
| het_suwa_meer_in_de_provincie_shinano_shinshu_su | 0.010 | 0.04 | 7 | 0.58 | 17.4 | 0.29 | S<0.35 cmp<0.2 |
| ono_shindon_in_the_suraga_province | 0.009 | 0.08 | 8 | 0.45 | 12.5 | 0.27 | S<0.35 cmp<0.2 nreg>7 |
| tsukudajima_in_musashi_province_buy_tsukudajima_ | 0.009 | 0.09 | 10 | 0.52 | 11.3 | 0.19 | S<0.35 cmp<0.2 nreg>7 |
| the_inume_pass_in_kai_province_ksh_inume_tge_fro | 0.008 | 0.06 | 7 | 0.51 | 10.7 | 0.26 | S<0.35 cmp<0.2 |
| mishima_pass_in_kai_province_ksh_mishima_goe_fro | 0.007 | 0.06 | 7 | 0.45 | 12.0 | 0.27 | S<0.35 cmp<0.2 |
| hodogaya_on_the_tkaid_tkaid_hodogaya_from_the_se | 0.007 | 0.05 | 7 | 0.51 | 10.6 | 0.32 | S<0.35 cmp<0.2 |
| fujithe_tama_river_musashi_province_from_the_ser | 0.006 | 0.09 | 8 | 0.40 | 9.2 | 0.23 | S<0.35 cmp<0.2 nreg>7 |
| umegawa_in_sagami_province | 0.006 | 0.04 | 9 | 0.47 | 17.9 | 0.24 | S<0.35 cmp<0.2 nreg>7 |