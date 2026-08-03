"""Optimiser bake-off over the joint design space, at equal evaluation budget.

Five search strategies are given the SAME space, the SAME objective, the SAME evaluation
budget and the SAME memoised cache, so the comparison is between the strategies and nothing
else. That last point matters more than it looks: every method here draws from one shared
journal, so a point already evaluated by an earlier method costs the later ones nothing --
which would quietly hand a late-running method a bigger effective budget. Budget is therefore
counted in DISTINCT evaluations, not in calls.

    python pearl3_search.py --budget 25 --methods random anneal cmaes tpe ga

Objective is the composite in `pearl3_sweep.Sweep`: a quarter each on mean and worst-view
IoU and a half on SSIM. Ranking on IoU alone would let a search win by nailing the silhouette
while the tone inside it drifts arbitrarily far off, which is exactly the failure the earlier
sweeps ran into.

Space parameterisation. The three engraved tones are searched as a lead level and two
RATIOS rather than as three free numbers, so every sample is monotonically darkening by
construction; sampling them independently would spend most of the budget on invalid orderings
and the comparison would then be measuring each method's rejection rate instead of its search.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from pearl3_baseline import ARMS
from pearl3_sweep import Sweep

# name, low, high, is_integer
SPACE: List[Tuple[str, float, float, bool]] = [
    ("n_per_family", 3, 8, True),
    ("pitch", 0.012, 0.035, False),
    ("t1", 0.50, 0.90, False),          # lightest engraved tone
    ("r2", 0.35, 0.80, False),          # t2 = t1 * r2
    ("r3", 0.25, 0.80, False),          # t3 = t2 * r3
    ("fragment_size", 0.04, 0.13, False),
    ("intensity_gain", 0.80, 1.10, False),
    ("detail_bias", 1.0, 3.5, False),
    ("grid_phase", 0.0, 60.0, False),   # rigid rotation of the whole family fan
]


def decode(x: Sequence[float], stops_deg) -> Dict:
    """Unit-cube point -> build overrides. Every method searches in [0,1]^d."""
    v = {}
    for xi, (name, lo, hi, is_int) in zip(x, SPACE):
        t = lo + float(np.clip(xi, 0.0, 1.0)) * (hi - lo)
        v[name] = int(round(t)) if is_int else float(t)
    t1 = v.pop("t1"); t2 = t1 * v.pop("r2"); t3 = t2 * v.pop("r3")
    phase = v.pop("grid_phase")
    v["engrave_levels"] = (round(t1, 4), round(t2, 4), round(t3, 4))
    # The stops stay evenly spaced -- only the fan's phase moves, which is the one rotation
    # freedom left open. Rounding keeps the cache key from fragmenting on meaningless digits.
    v["family_angles_deg"] = tuple(round(((270.0 - s) + phase) % 180.0, 2) for s in stops_deg)
    return v


class BudgetExhausted(BaseException):
    """Raised to stop a strategy the moment its budget is spent.

    A BaseException, not an Exception, on purpose: several of these libraries wrap the
    objective in a broad `except Exception` to survive user errors, and a budget stop that
    gets swallowed there would silently hand that method an unlimited budget and invalidate
    the whole comparison.
    """


def make_objective(sw: Sweep, stops_deg, budget: int):
    """Returns (f, state). `f` maximises the composite score and enforces the budget."""
    state = {"n": 0, "best": -1e9, "best_ov": None, "history": []}

    def f(x: Sequence[float]) -> float:
        ov = decode(x, stops_deg)
        fresh = json.dumps(ov, sort_keys=True, default=str) not in sw.cache
        if fresh and state["n"] >= budget:
            raise BudgetExhausted
        try:
            rec = sw.evaluate(ov)
        except (AssertionError, ValueError):
            # Infeasible geometry (e.g. the sheet stack alone overruns the footprint).
            # A finite penalty rather than a raise, so the gradient-free methods learn the
            # region is bad instead of the run dying on a sample that was always going to
            # be discarded.
            return -1.0
        if fresh:
            state["n"] += 1
        s = rec["score"]
        if s > state["best"]:
            state["best"], state["best_ov"] = s, ov
        state["history"].append((state["n"], s, state["best"]))
        return s

    return f, state


# --- the five strategies ----------------------------------------------------
def run_random(f, d, budget, rng):
    """Uniform random search. The control arm: any method that cannot beat this is noise."""
    while True:
        f(rng.random(d))


def run_anneal(f, d, budget, rng):
    """Simulated annealing with a geometric cooling schedule and a shrinking step.

    Included because the repo's earlier logo work found annealing beat hill climbing while
    using fewer evaluations -- it escapes the separated basins a greedy climb gets stuck in.
    The step shrinks with temperature so late moves are local polish, not fresh restarts.
    """
    x = rng.random(d)
    fx = f(x)
    T0, T1 = 0.25, 0.005
    for i in range(10 ** 6):
        T = T0 * (T1 / T0) ** min(i / max(budget - 2, 1), 1.0)
        y = np.clip(x + rng.normal(0, T, d), 0, 1)
        fy = f(y)
        if fy > fx or rng.random() < math.exp((fy - fx) / max(T, 1e-6)):
            x, fx = y, fy


def run_cmaes(f, d, budget, rng):
    """CMA-ES: adapts a full covariance, so it can follow a diagonal valley in the space.

    The design axes here are strongly coupled -- tighter pitch wants more sheets, darker
    tones want a lower intensity gain -- and an axis-aligned method has to zig-zag along
    such a valley. This is the one method in the set that learns the coupling.
    """
    import cma
    es = cma.CMAEvolutionStrategy([0.5] * d, 0.25,
                                  {"bounds": [0, 1], "verbose": -9, "seed": 1})
    while True:
        xs = es.ask()
        es.tell(xs, [-f(x) for x in xs])       # cma minimises


def run_tpe(f, d, budget, rng):
    """Optuna's TPE: models good vs bad regions densely, strong when evaluations are costly."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=1))
    st.optimize(lambda t: f([t.suggest_float(f"x{i}", 0.0, 1.0) for i in range(d)]),
                n_trials=10 ** 6)


def run_ga(f, d, budget, rng):
    """Generational GA: blend crossover + Gaussian mutation, tournament selection, elitism.

    Crossover is the thing none of the others do -- it recombines whole partial designs (a
    good layout from one individual with a good tone ladder from another) rather than
    perturbing a single point, which is the natural move when the space is a bag of loosely
    coupled subsystems.
    """
    pop_n = max(6, budget // 5)
    pop = [rng.random(d) for _ in range(pop_n)]
    fit = [f(p) for p in pop]
    while True:
        children = []
        for _ in range(pop_n):
            pa, pb = (max(rng.choice(len(pop), 2, replace=False), key=lambda i: fit[i])
                      for _ in range(2))
            w = rng.random(d)                                   # blend crossover
            children.append(np.clip(w * pop[pa] + (1 - w) * pop[pb]
                                    + rng.normal(0, 0.08, d), 0, 1))
        cf = [f(c) for c in children]
        keep = sorted(zip(fit + cf, pop + children), key=lambda t: -t[0])[:pop_n]
        fit = [k[0] for k in keep]
        pop = [k[1] for k in keep]


METHODS: Dict[str, Callable] = {
    "random": run_random, "anneal": run_anneal,
    "cmaes": run_cmaes, "tpe": run_tpe, "ga": run_ga,
}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), default="30v4")
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    base = ARMS[a.arm]
    out = Path(a.out or f"out_pearl3/search_{a.arm}")
    out.mkdir(parents=True, exist_ok=True)
    d = len(SPACE)

    results = {}
    for name in a.methods:
        sw = Sweep(base, out, range(a.seeds))      # one shared journal across all methods
        f, state = make_objective(sw, base.stops_deg, a.budget)
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        try:
            METHODS[name](f, d, a.budget, rng)
        except BudgetExhausted:
            pass
        dt = time.perf_counter() - t0
        results[name] = {"best": state["best"], "overrides": state["best_ov"],
                         "n_evals": state["n"], "seconds": dt,
                         "history": state["history"]}
        print(f"{name:8s} best score {state['best']:.4f} after {state['n']} distinct evals "
              f"({dt:.0f}s)")

    (out / "bakeoff.json").write_text(json.dumps(results, indent=2, default=str),
                                      encoding="utf-8")
    print("\n  === bake-off, equal budget ===")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["best"]):
        print(f"  {name:8s} {r['best']:.4f}   {r['overrides']}")
    print(f"\n-> {out/'bakeoff.json'}")


if __name__ == "__main__":
    main()
