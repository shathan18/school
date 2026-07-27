"""Full 300-shard render for the 3 gate-2-curated candidates: red_fuji x tempesta (Hokusai's own
calm/storm Fuji duality), wpa_lassen x wpa_yellowstone (same 1938 federal campaign, both
volcanic/geothermal plumes), kajikazawa x at_sea_off_kazusa (same series, coastal/maritime theme).
Reuses render_pair.render_pair() unmodified."""
import json
from pathlib import Path
import render_pair as RP

PAIRS = [
    ("red_fuji__tempesta_storm", "examples/red_fuji.jpg",
     "examples/series/katsushika_hokusai_tempesta_sotto_la_vetta_dalla.jpg"),
    ("lassen__yellowstone_wpa", "examples/new_candidates/wpa_lassen_volcanic_1938.jpg",
     "examples/new_candidates/wpa_yellowstone_1938.jpg"),
    ("kajikazawa__at_sea_kazusa", "examples/kajikazawa.jpg",
     "examples/series/at_sea_off_kazusa_kazusa_no_kairo_from_the_serie.jpg"),
]

if __name__ == "__main__":
    RP.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for label, pa, pb in PAIRS:
        m = RP.render_pair(pa, pb, label)
        results.append(m)

    lines = [
        "# V3 gate-2-curated pair render results (300-shard, lab-gated)",
        "",
        f"Recipe: PC={RP.PANEL_COUNT}, seed={RP.SEED}, dw={RP.DAMAGE_WEIGHT}, cw={RP.CREDIT_WEIGHT}, "
        f"match_metric='{RP.MATCH_METRIC}', solver_tol={RP.MATCH_TOL_SOLVER}, report_tol={RP.MATCH_TOL_REPORT}.",
        "",
        "| label | SSIM A/B | edge A/B | good A/B | bad A/B | good/bad | shards |",
        "|-------|:--------:|:--------:|:--------:|:-------:|:--------:|:------:|",
    ]
    for r in results:
        gb = f"{r['good_bad_ratio']:.2f}" if r["good_bad_ratio"] is not None else "inf"
        lines.append(f"| {r['label']} | {r['ssim_A']:.3f}/{r['ssim_B']:.3f} | "
                      f"{r['edge_A']:.3f}/{r['edge_B']:.3f} | {r['good_A']:.1f}/{r['good_B']:.1f}% | "
                      f"{r['bad_A']:.1f}/{r['bad_B']:.1f}% | {gb} | {r['n_shards']} |")
    Path("out_pair_selection/v3_pair_render_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nwrote out_pair_selection/v3_pair_render_summary.md")
