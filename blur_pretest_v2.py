"""Extended blur pre-test, round 2: original pool + round-1 new candidates (Hiroshige/Mondrian/
WPA/Friedrich) + round-2 new candidates (Hiroshige snow duo, WPA Zion) -- all sourced in
CONNECTED GROUPS this round (see plan) so gate 2 is satisfied by construction, not by blind
palette matching. Reuses blur_pretest.py's scoring/output machinery unmodified."""
from pathlib import Path
import blur_pretest as BP

_original_candidate_pool = BP.candidate_pool


def extended_pool() -> list[Path]:
    pool = _original_candidate_pool()
    for d in ("examples/new_candidates", "examples/new_candidates_v2"):
        new = sorted(p for p in Path(d).glob("*.jpg"))
        seen = {p.stem for p in pool}
        for p in new:
            if p.stem not in seen:
                pool.append(p); seen.add(p.stem)
    return pool


if __name__ == "__main__":
    BP.candidate_pool = extended_pool
    BP.main()
