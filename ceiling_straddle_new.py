"""ceiling_straddle analysis (reusing out_thickness_test/ceiling_straddle.py's functions
unmodified, just fixed the stale sys.path and swapped in the new candidate pairs) on the
standout new-style pairs: fuji_a x wpa_yellowstone, mondrian_ii x mondrian_iii, plus the
reference dawn_isawa x goten_yama for comparison."""
import sys, os, importlib.util
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("cs", "out_thickness_test/ceiling_straddle.py")
cs = importlib.util.module_from_spec(spec)
# the module runs its own PAIRS demo at import time via top-level code guarded by no __main__
# check -- it does NOT: everything after PAIRS=[...] is under no guard, so importing would
# execute the OLD apples/monet/mona demo. Patch: exec only up to (not including) the PAIRS block.
src = open("out_thickness_test/ceiling_straddle.py", encoding="utf-8").read()
src = src.split("PAIRS = [")[0]
src = src.replace(r'sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")', "")
ns = {}
exec(compile(src, "ceiling_straddle_lib", "exec"), ns)
build, achieved_bgood, ceilings_and_straddle = ns["build"], ns["achieved_bgood"], ns["ceilings_and_straddle"]

PAIRS = [
    ("red_fuji x tempesta storm (V3 lead)", "examples/red_fuji.jpg",
     "examples/series/katsushika_hokusai_tempesta_sotto_la_vetta_dalla.jpg"),
    ("lassen x yellowstone WPA (V3 runner-up)", "examples/new_candidates/wpa_lassen_volcanic_1938.jpg",
     "examples/new_candidates/wpa_yellowstone_1938.jpg"),
    ("kajikazawa x at_sea_kazusa (V3 rejected)", "examples/kajikazawa.jpg",
     "examples/series/at_sea_off_kazusa_kazusa_no_kairo_from_the_serie.jpg"),
]

print(f"{'pair':38s} {'wall':5s} {'B_good':>8s} {'ceil(a)':>8s} {'ceil(c)':>8s} {'straddle':>9s}")
print("-" * 82)
for label, pa, pb in PAIRS:
    ts, table, t, panels = build(pa, pb, seed=1)
    ach = achieved_bgood(ts, table, t, panels)
    csr = ceilings_and_straddle(ts, table, t, panels)
    for w in ("A", "B"):
        c = csr.get(w, {})
        print(f"{label if w=='A' else '':38s} {w:5s} {ach[w]:7.1f}% "
              f"{c.get('ceil_a',float('nan')):7.1f}% {c.get('ceil_c',float('nan')):7.1f}% "
              f"{c.get('straddle',float('nan'))*100:8.0f}%")
    print()
print("legend: B_good = achieved honest colour-agreeing coverage of secondary subject (current pipeline).")
print("        ceil(a) = best achievable by re-choosing depth only (colour fixed from wall A).")
print("        ceil(c) = best achievable with free compromise colour (option c upside).")
print("        straddle = colour-uniformity of landing blobs (100%=shape not the cap; low=shape caps c).")
