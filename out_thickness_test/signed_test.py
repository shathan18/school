"""HYPOTHESIS TEST: does SIGNED damage recover joint-intersection (B_good) while keeping
bad cross-talk (B_bad) low -- or do they move in lockstep (=> trade-off is fundamental)?

Arms: control (rng.choice) | unsigned damage (harm-only) | signed damage x 3 credit weights.
Same 10 seeds, same fixed panel placement (wide/seed), 0.5x density. Placement UNCHANGED."""
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

scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res
names=["clear"]+C.CMYK
targets={"A":C.load_color_target("examples/apples.jpg",wr,white_thr=scene.white_threshold),
         "B":C.load_color_target("examples/breakfast.jpg",wr,white_thr=scene.white_threshold)}
subject={w:C.subject_mask(targets[w],scene.white_threshold) for w in ("A","B")}
FS=0.135/np.sqrt(0.5)
SP=dataclasses.replace(scene.solve,fragment_size=FS,
   fragment_min_area=scene.solve.fragment_min_area*(FS/0.135)**2,
   fragment_max_area=scene.solve.fragment_max_area*(FS/0.135)**2)
SEEDS=list(range(1,11))

# (label, damage_weight, credit_weight)
ARMS=[("control (rng.choice)",0.0,None),
      ("unsigned (harm-only)",0.5,None),
      ("signed credit=0.5",0.5,0.5),
      ("signed credit=1.0",0.5,1.0),
      ("signed credit=2.0",0.5,2.0)]

def evaluate(panels,seed,dw,cw):
    ts=dataclasses.replace(scene,panels=panels,solve=SP)
    table=build_projection_table(ts); renderer=Renderer(ts,table)
    sc,op,fr,res,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,
        white_thr=ts.white_threshold,max_stack=ts.color_max_stack,seed=seed,
        damage_weight=dw,credit_weight=cw)
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
        comp[w]=100*(d&subject[w]).sum()/max(subject[w].sum(),1)
        nonprim={p.name for p in panels if prim[p.name]!=w}
        xt=dark(rk(nonprim)[w]) if nonprim else np.zeros_like(d)
        ct[w]={"bad":100*(xt&~subject[w]).sum()/max(tot,1),
               "good":100*(xt&subject[w]).sum()/max(tot,1)}
    used=len({f["panel"] for f in fr})
    return acc,ct,comp,used,len(panels)

results=[]
for label,dw,cw in ARMS:
    print(f"\n--- {label} ---")
    for seed in SEEDS:
        panels,_=build_panels_greedy(scene,count=14,mode="deliberate",K=16,targets=targets,
                                     seed=seed,angle_deg_range=(5,85))   # placement UNCHANGED
        acc,ct,comp,used,npan=evaluate(panels,seed,dw,cw)
        r={"arm":label,"dw":dw,"cw":cw,"seed":seed,"used":used,"npanels":npan,
           "compA":comp["A"],"compB":comp["B"],
           "A_bad":ct["A"]["bad"],"B_bad":ct["B"]["bad"],
           "A_good":ct["A"]["good"],"B_good":ct["B"]["good"],
           "A_rmse":acc["A"]["rmse"],"A_ssim":acc["A"]["ssim"],
           "B_rmse":acc["B"]["rmse"],"B_ssim":acc["B"]["ssim"]}
        results.append(r)
        print(f"  seed={seed:2d} panels={used:2d}/{npan} comp={comp['A']:5.1f}/{comp['B']:5.1f}% | "
              f"B_bad={r['B_bad']:5.1f}% B_good={r['B_good']:5.1f}% | "
              f"A {acc['A']['rmse']:.3f}/{acc['A']['ssim']:.3f} B {acc['B']['rmse']:.3f}/{acc['B']['ssim']:.3f}")

json.dump(results,open("out_thickness_test/signed_results.json","w"),indent=1)

print("\n"+"="*100)
print("HEADLINE: do B_good and B_bad move TOGETHER (fundamental trade-off) or SEPARATE?")
print("="*100)
print(f"{'arm':22s} | {'B_bad':>7s} {'B_good':>7s} | {'A_ssim':>7s} {'B_ssim':>7s} {'A_rmse':>7s} {'B_rmse':>7s} | {'panels':>7s} {'minComp':>8s}")
for label,dw,cw in ARMS:
    sub=[r for r in results if r["arm"]==label]
    print(f"{label:22s} | {st.mean(r['B_bad'] for r in sub):6.1f}% {st.mean(r['B_good'] for r in sub):6.1f}% | "
          f"{st.mean(r['A_ssim'] for r in sub):7.3f} {st.mean(r['B_ssim'] for r in sub):7.3f} "
          f"{st.mean(r['A_rmse'] for r in sub):7.3f} {st.mean(r['B_rmse'] for r in sub):7.3f} | "
          f"{st.mean(r['used'] for r in sub):5.1f}/14 {min(min(r['compA'],r['compB']) for r in sub):7.1f}%")
print("\nwrote out_thickness_test/signed_results.json")
