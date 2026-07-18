"""
Does running the greedy MORE times actually help?

Each restart is one seed (one `build_panels_greedy` layout + one `fragment_shards_overlap`
placement), and the deliverable keeps the best by mean RMSE. So best-of-5 / best-of-10 /
best-of-20 are simply the best over seeds 1-5 / 1-10 / 1-20 of a single 20-seed run -- nested
subsets of identical seeds, which isolates the restart effect with no seed-luck confound
(and costs 20 solves per pairing instead of 35).

Reads each run's metrics.json `per_seed` list and reports, per pairing, what each restart
budget would have bought.

Run:  py out_thickness_test/restart_compare.py
"""
import os, json
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

RUNS = [("vase x surface",   "out_thickness_test/mix_vase_surface"),
        ("vase x oranges",   "out_thickness_test/mix_vase_oranges"),
        ("surface x oranges","out_thickness_test/mix_surface_oranges")]
BUDGETS = [5, 10, 20, 50, 100]
OUT = "out_thickness_test/mona_pairs"; os.makedirs(OUT, exist_ok=True)


def best_of(per_seed, n):
    """Same selection rule the deliverable uses: lowest mean RMSE over the first n seeds."""
    sub = [s for s in per_seed if s["seed"] <= n]
    if not sub:
        return None
    b = min(sub, key=lambda s: 0.5 * (s["A"]["rmse"] + s["B"]["rmse"]))
    return dict(seed=b["seed"], ssimA=b["A"]["ssim"], ssimB=b["B"]["ssim"],
                clarity=b["A"]["ssim"] + b["B"]["ssim"],
                edgeA=b["A"]["edge_fidelity"], edgeB=b["B"]["edge_fidelity"],
                rmse=0.5 * (b["A"]["rmse"] + b["B"]["rmse"]))


table = {}
print(f"{'pairing':20s} {'N':>4} {'best seed':>10} {'SSIM A/B':>14} {'clarity':>8} {'mean RMSE':>10}")
print("-" * 74)
for label, d in RUNS:
    p = os.path.join(d, "metrics.json")
    if not os.path.exists(p):
        print(f"  MISSING {d} -- skipped"); continue
    with open(p) as f:
        m = json.load(f)
    per_seed = m.get("per_seed", [])
    if not per_seed:
        print(f"  {d}: no per_seed data"); continue
    table[label] = {}
    have = max(s["seed"] for s in per_seed)
    for n in BUDGETS:
        if n > have:            # don't report a budget we didn't actually run
            continue
        b = best_of(per_seed, n)
        if b is None:
            continue
        table[label][n] = b
        print(f"{label:20s} {n:>4} {b['seed']:>10} {b['ssimA']:.3f}/{b['ssimB']:.3f}   "
              f"{b['clarity']:.3f}   {b['rmse']:.4f}")
    print("-" * 74)

# verdict: did more restarts actually buy anything?
print("\nGAIN from extra restarts (clarity):")
for label, byn in table.items():
    if 5 in byn and 20 in byn:
        d10 = byn.get(10, byn[5])["clarity"] - byn[5]["clarity"]
        d20 = byn[20]["clarity"] - byn[5]["clarity"]
        print(f"  {label:20s} 5->10 {d10:+.4f}   5->20 {d20:+.4f}"
              f"   (best seed moved {byn[5]['seed']} -> {byn[20]['seed']})")

if table:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, byn in table.items():
        ns = sorted(byn); ax[0].plot(ns, [byn[n]["clarity"] for n in ns], "o-", label=label)
        ax[1].plot(ns, [byn[n]["rmse"] for n in ns], "o-", label=label)
    ax[0].set_xlabel("greedy restarts (N)"); ax[0].set_ylabel("clarity (SSIM_A+SSIM_B)")
    ax[0].set_title("Higher is better"); ax[0].set_xticks(BUDGETS)
    ax[1].set_xlabel("greedy restarts (N)"); ax[1].set_ylabel("mean RMSE (selection metric)")
    ax[1].set_title("Lower is better"); ax[1].set_xticks(BUDGETS)
    for a in ax: a.grid(alpha=0.3); a.legend(fontsize=8)
    plt.suptitle("Does running the greedy more times help? (nested best-of-N, same seeds)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT}/restart_compare.png", dpi=110, bbox_inches="tight"); plt.close()
    print(f"\nsaved {OUT}/restart_compare.png")
