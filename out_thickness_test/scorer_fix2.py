"""Multiplicative subject-aware scorer: score = gain * (1 - spill_frac)^lambda.
Control = lambda 0 (unfixed). 10 seeds each. 0.5x density, wide angle.
BLANK-WALL GUARD: image-completeness < 85% on either wall => AUTO-DISQUALIFIED."""
import sys, dataclasses, json
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np, statistics as st
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics

COMPLETENESS_MIN = 85.0        # hard guard: below this on EITHER wall = disqualified

scene = load_scene("scenes/example.yaml"); wr=scene.solve.wall_res
names=["clear"]+C.CMYK
targets = {"A": C.load_color_target("examples/apples.jpg", wr, white_thr=scene.white_threshold),
           "B": C.load_color_target("examples/breakfast.jpg", wr, white_thr=scene.white_threshold)}
subject={w:C.subject_mask(targets[w],scene.white_threshold) for w in ("A","B")}
FS = 0.135/np.sqrt(0.5)
SP = dataclasses.replace(scene.solve, fragment_size=FS,
     fragment_min_area=scene.solve.fragment_min_area*(FS/0.135)**2,
     fragment_max_area=scene.solve.fragment_max_area*(FS/0.135)**2)
SEEDS=list(range(1,11)); LAMBDAS=[0.0,1.0,2.0,4.0]

def evaluate(panels, seed):
    ts=dataclasses.replace(scene,panels=panels,solve=SP)
    table=build_projection_table(ts); renderer=Renderer(ts,table)
    sc,op,fr,res,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,
        white_thr=ts.white_threshold,max_stack=ts.color_max_stack,seed=seed)
    panel_T=C.stack_transmit_lut(names,sc,si)
    pred=renderer.render_color_np(panel_T)
    acc=_metrics.evaluate_wall_accuracy(targets,pred)
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}
    def rk(keep):
        pt=panel_T.copy()
        for gi,p in enumerate(panels):
            if p.name not in keep: pt[gi]=1.0
        return renderer.render_color_np(pt)
    dark=lambda im:(1.0-im.mean(axis=-1))>0.05
    ct={}; comp={}
    for w in ("A","B"):
        d=dark(pred[w]); tot=d.sum()
        comp[w]=100*(d&subject[w]).sum()/max(subject[w].sum(),1)     # BLANK-WALL GUARD input
        nonprim={p.name for p in panels if prim[p.name]!=w}
        xt=dark(rk(nonprim)[w]) if nonprim else np.zeros_like(d)
        ct[w]={"bad":100*(xt&~subject[w]).sum()/max(tot,1),
               "good":100*(xt&subject[w]).sum()/max(tot,1)}
    return acc, ct, comp

results=[]
for lam in LAMBDAS:
    for seed in SEEDS:
        panels,_=build_panels_greedy(scene,count=14,mode="deliberate",K=16,targets=targets,
                                     seed=seed, angle_deg_range=(5,85), spill_weight=lam)
        acc,ct,comp=evaluate(panels,seed)
        dq = comp["A"]<COMPLETENESS_MIN or comp["B"]<COMPLETENESS_MIN
        r={"lam":lam,"seed":seed,"compA":comp["A"],"compB":comp["B"],"disqualified":bool(dq),
           "A_bad":ct["A"]["bad"],"B_bad":ct["B"]["bad"],
           "A_good":ct["A"]["good"],"B_good":ct["B"]["good"],
           "A_rmse":acc["A"]["rmse"],"A_ssim":acc["A"]["ssim"],"A_edge":acc["A"]["edge_fidelity"],
           "B_rmse":acc["B"]["rmse"],"B_ssim":acc["B"]["ssim"],"B_edge":acc["B"]["edge_fidelity"]}
        results.append(r)
        tag=" *** DISQUALIFIED (blank wall) ***" if dq else ""
        print(f"lam={lam:<4} seed={seed:2d} | completeness A={comp['A']:5.1f}% B={comp['B']:5.1f}% | "
              f"B_bad={r['B_bad']:5.1f}% B_good={r['B_good']:5.1f}% | "
              f"A {acc['A']['rmse']:.3f}/{acc['A']['ssim']:.3f} B {acc['B']['rmse']:.3f}/{acc['B']['ssim']:.3f}{tag}")

json.dump(results,open("out_thickness_test/scorer_fix2_results.json","w"),indent=1)

print(f"\n=== SUMMARY (guard: completeness >= {COMPLETENESS_MIN}% on BOTH walls) ===")
print("Control robustness to beat: B_bad stdev=7.1, 2/10 seeds <15%\n")
for lam in LAMBDAS:
    sub=[r for r in results if r["lam"]==lam]
    ok=[r for r in sub if not r["disqualified"]]
    dq=len(sub)-len(ok)
    tag=" (CONTROL)" if lam==0 else ""
    print(f"lam={lam}{tag}: {len(ok)}/10 seeds pass the completeness guard ({dq} disqualified)")
    if not ok:
        print("   -> NO valid configs. Any low bad-cross-talk here is a BLANK WALL, not a win.\n")
        continue
    bb=[r["B_bad"] for r in ok]
    print(f"   among PASSING seeds only: B_bad mean={st.mean(bb):5.1f}% median={st.median(bb):5.1f}% "
          f"min={min(bb):5.1f}% max={max(bb):5.1f}% stdev={st.pstdev(bb):4.1f} | <15%: {sum(1 for x in bb if x<15)}/{len(ok)}")
    print(f"   B_good mean={st.mean(r['B_good'] for r in ok):5.1f}%  "
          f"A_rmse={st.mean(r['A_rmse'] for r in ok):.3f} A_ssim={st.mean(r['A_ssim'] for r in ok):.3f} "
          f"B_rmse={st.mean(r['B_rmse'] for r in ok):.3f} B_ssim={st.mean(r['B_ssim'] for r in ok):.3f}\n")
print("wrote out_thickness_test/scorer_fix2_results.json")
