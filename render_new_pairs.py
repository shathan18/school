"""Full 300-shard render for the extended-pool's most interesting pairs: the existing #1
(reference) plus three genuinely NEW cross-movement pairs surfaced by the extended survivor
pool (Hokusai x Mondrian, bold-poster x bold-poster across cultures, and a same-artist
Mondrian pair). Reuses render_pair.render_pair() unmodified."""
import json
from pathlib import Path
import render_pair as RP

PAIRS = [
    ("dawn_isawa__goten_yama", "examples/series/dawn_at_isawa_in_the_kai_province.jpg",
     "examples/series/katsushika_hokusai_goten_yama_hill_shinagawa_on_.jpg"),
    ("dawn_isawa__mondrian_iii", "examples/series/dawn_at_isawa_in_the_kai_province.jpg",
     "examples/new_candidates/mondrian_comp_iii_1929.jpg"),
    ("fuji_a__wpa_yellowstone", "examples/fuji_a.jpg",
     "examples/new_candidates/wpa_yellowstone_1938.jpg"),
    ("mondrian_ii__mondrian_iii", "examples/new_candidates/mondrian_comp_ii_red_blue_yellow_1930.jpg",
     "examples/new_candidates/mondrian_comp_iii_1929.jpg"),
]

if __name__ == "__main__":
    RP.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for label, pa, pb in PAIRS:
        m = RP.render_pair(pa, pb, label)
        results.append(m)

    results.sort(key=lambda r: -(r["good_mean"] - r["bad_mean"]))
    lines = [
        "# New-style pair render results (300-shard, lab-gated)",
        "",
        f"Recipe: PC={RP.PANEL_COUNT}, seed={RP.SEED}, dw={RP.DAMAGE_WEIGHT}, cw={RP.CREDIT_WEIGHT}, "
        f"match_metric='{RP.MATCH_METRIC}', solver_tol={RP.MATCH_TOL_SOLVER}, report_tol={RP.MATCH_TOL_REPORT}.",
        "",
        "| # | label | SSIM A/B | edge A/B | good A/B | bad A/B | good/bad | shards | elapsed |",
        "|---|-------|:--------:|:--------:|:--------:|:-------:|:--------:|:------:|:-------:|",
    ]
    for i, r in enumerate(results, 1):
        gb = f"{r['good_bad_ratio']:.2f}" if r["good_bad_ratio"] is not None else "inf"
        lines.append(f"| {i} | {r['label']} | {r['ssim_A']:.3f}/{r['ssim_B']:.3f} | "
                      f"{r['edge_A']:.3f}/{r['edge_B']:.3f} | {r['good_A']:.1f}/{r['good_B']:.1f}% | "
                      f"{r['bad_A']:.1f}/{r['bad_B']:.1f}% | {gb} | {r['n_shards']} | {r['elapsed_s']}s |")
    Path("out_pair_selection/new_pair_render_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nwrote out_pair_selection/new_pair_render_summary.md")
