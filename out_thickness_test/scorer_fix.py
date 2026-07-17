"""Subject-aware scorer fix: penalize outside-subject cross-talk during candidate selection.
Control arm = spill_weight 0 (the unfixed scorer). 10 seeds each. 0.5x density, wide angle."""
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

scene = load_scene("scenes/example.yaml"); wr=scene.solve.wall_res
names=["clear"]+C.CMYK
targets = {"A": C.load_color_target("examples/apples.jpg", wr, white_thr=scene.white_threshold),
           "B": C.load_color_target("examples/breakfast.jpg", wr, white_thr=scene.white_threshold)}
subject={w:C.subject_mask(targets[w],scene.white_threshold) for w in ("A","B")}

# 0.5x density = confirmed structural fidelity sweet spot
FS = 0.135/np.sqrt(0.5)
SP = dataclasses.replace(scene.solve, fragment_size=FS,
     fragment_min_area=scene.solve.fragment_min_area*(FS/0.135)**2,
     fragment_max_area=scene.solve.fragment_max_area*(FS/0.135)**2)
SEEDS = list(range(1,11))
LAMBDAS = [0.0, 1.0, 2.0, 4.0]     # 0.0 = control (unfixed scorer)

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
    ct={}
    for w in ("A","B"):
        nonprim={p.name for p in panels if prim[p.name]!=w}
        tot=dark(pred[w]).sum()
        xt=dark(rk(nonprim)[w]) if nonprim else np.zeros_like(dark(pred[w]))
        ct[w]={"bad":100*(xt&~subject[w]).sum()/max(tot,1),
               "good":100*(xt&subject[w]).sum()/max(tot,1)}
    a=sum(1 for p in panels if prim[p.name]=="A")
    return acc, ct, a, len(panels)

results=[]
for lam in LAMBDAS:
    for seed in SEEDS:
        panels,_=build_panels_greedy(scene,count=14,mode="deliberate",K=16,targets=targets,
                                     seed=seed, angle_deg_range=(5,85), spill_weight=lam)
        acc,ct,a,n=evaluate(panels,seed)
        r={"lam":lam,"seed":seed,"panels":n,"split_A":a,"split_B":n-a,
           "A_bad":ct["A"]["bad"],"B_bad":ct["B"]["bad"],
           "A_good":ct["A"]["good"],"B_good":ct["B"]["good"],
           "A_rmse":acc["A"]["rmse"],"A_ssim":acc["A"]["ssim"],"A_edge":acc["A"]["edge_fidelity"],
           "B_rmse":acc["B"]["rmse"],"B_ssim":acc["B"]["ssim"],"B_edge":acc["B"]["edge_fidelity"]}
        results.append(r)
        print(f"lam={lam:<4} seed={seed:2d} n={n:2d} split {a}/{n-a} | "
              f"B_bad={r['B_bad']:5.1f}% B_good={r['B_good']:5.1f}% A_bad={r['A_bad']:4.1f}% | "
              f"A {acc['A']['rmse']:.3f}/{acc['A']['ssim']:.3f}/{acc['A']['edge_fidelity']:.3f} "
              f"B {acc['B']['rmse']:.3f}/{acc['B']['ssim']:.3f}/{acc['B']['edge_fidelity']:.3f}")

json.dump(results,open("out_thickness_test/scorer_fix_results.json","w"),indent=1)
print("\n=== SUMMARY per lambda (across 10 seeds) ===")
for lam in LAMBDAS:
    sub=[r for r in results if r["lam"]==lam]
    bb=[r["B_bad"] for r in sub]; bg=[r["B_good"] for r in sub]
    tag=" (CONTROL = unfixed scorer)" if lam==0 else ""
    print(f"lam={lam}{tag}:")
    print(f"   B_bad : mean={st.mean(bb):5.1f}%  median={st.median(bb):5.1f}%  "
          f"min={min(bb):5.1f}%  max={max(bb):5.1f}%  stdev={st.pstdev(bb):4.1f}  "
          f"seeds<15%: {sum(1 for x in bb if x<15)}/10")
    print(f"   B_good: mean={st.mean(bg):5.1f}%   |  "
          f"A_ssim mean={st.mean(r['A_ssim'] for r in sub):.3f}  B_ssim mean={st.mean(r['B_ssim'] for r in sub):.3f}  "
          f"A_rmse mean={st.mean(r['A_rmse'] for r in sub):.3f}  B_rmse mean={st.mean(r['B_rmse'] for r in sub):.3f}")
print("\nwrote out_thickness_test/scorer_fix_results.json")
