"""Extended blur pre-test: teammate's original candidate_pool() (Hokusai series + bold Fuji
cuts) PLUS new candidates spanning Hiroshige, Mondrian, WPA Art Deco posters, and Caspar David
Friedrich (see examples/new_candidates/_manifest.txt for sourcing). Reuses blur_pretest.py's
scoring/output machinery unmodified -- only the candidate pool is extended, per the same
survivability criterion (single bold shape, few colours, survives heavy blur)."""
from pathlib import Path
import blur_pretest as BP


_original_candidate_pool = BP.candidate_pool   # capture BEFORE any monkeypatch


def extended_pool() -> list[Path]:
    pool = _original_candidate_pool()
    new_dir = Path("examples/new_candidates")
    new = sorted(p for p in new_dir.glob("*.jpg"))
    seen = {p.stem for p in pool}
    for p in new:
        if p.stem not in seen:
            pool.append(p); seen.add(p.stem)
    return pool


if __name__ == "__main__":
    BP.candidate_pool = extended_pool   # monkeypatch: reuse BP.main() unchanged
    BP.main()
