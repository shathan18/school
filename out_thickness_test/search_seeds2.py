"""Higher-shard multi-seed: the 87-shard configs FAILED the 85% completeness gate on most
seeds. Re-test at more shards (still <=300 total) to find a config that RELIABLY passes."""
import sys, dataclasses, json, statistics as st
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics

A_IMG="examples/Screenshot 2026-07-16 102515.png"; B_IMG="examples/Screenshot 2026-07-16 102536.png"
scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res
names=["clear"]+C.CMYK
targets={"A":C.load_color_target(A_IMG,wr,white_thr=scene.white_threshold),
         "B":C.load_color_target(B_IMG,wr,white_thr=scene.white_threshold)}
subject={w:C.subject_mask(targets[w],scene.white_threshold) for w in ("A","B")}
def cryptic(ts,table,r,panels,pT):
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}; carry=[]
    for gi,p in enumerate(panels):
        w=prim[p.name]; pt=np.ones_like(pT); pt[gi]=pT[gi]
        d=(1.0-r.render_color_np(pt)[w].mean(-1))>0.05
        carry.append((d&subject[w]).sum()/max(subject[w].sum(),1))
    return max(carry)
def evaluate(pc,mult,seed):
    panels,_=build_panels_greedy(scene,count=pc,mode="deliberate",K=16,targets=targets,seed=seed,angle_deg_range=(5,85))
    FS=0.135/np.sqrt(mult)
    SP=dataclasses.replace(scene.solve,fragment_size=FS,
       fragment_min_area=scene.solve.fragment_min_area*(FS/0.135)**2,
       fragment_max_area=scene.solve.fragment_max_area*(FS/0.135)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=SP)
    table=build_projection_table(ts); r=Renderer(ts,table)
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,
        white_thr=ts.white_threshold,max_stack=ts.color_max_stack,seed=seed,
        damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0)
    pT=C.stack_transmit_lut(names,sc,si); pred=r.render_color_np(pT)
    acc=_metrics.evaluate_wall_accuracy(targets,pred)
    nA=bs.get("A",{}).get("achieved",0); nB=bs.get("B",{}).get("achieved",0)
    comp=[100*(((1.0-pred[w].mean(-1))>0.05)&subject[w]).sum()/subject[w].sum() for w in ("A","B")]
    prim=(acc["A"]["ssim"]+acc["B"]["ssim"]+acc["A"]["edge_fidelity"]+acc["B"]["edge_fidelity"])/2
    return prim, cryptic(ts,table,r,panels,pT), min(comp), nA+nB
CANDS=[(12,1.0),(16,1.0),(16,0.7),(20,1.0)]
SEEDS=[1,2,3,4,5,6]; results={}
for pc,mult in CANDS:
    P=[];Cy=[];Cp=[];T=[]
    print(f"\n--- panels={pc}, mult={mult} ---")
    for s in SEEDS:
        prim,carry,mincomp,tot=evaluate(pc,mult,s); P.append(prim);Cy.append(carry);Cp.append(mincomp);T.append(tot)
        print(f"  seed {s}: primary={prim:.3f} carry={carry:.3f} min-comp={mincomp:.0f}% shards={tot} {'' if (mincomp>=85 and tot<=300) else '<-- FAILS gate'}")
    passed=sum(1 for c,t in zip(Cp,T) if c>=85 and t<=300)
    results[f"{pc}_{mult}"]=dict(primary=P,carry=Cy,comp=Cp,total=T,passed=passed)
    print(f"  gate passes {passed}/6 seeds | MEAN primary={st.mean(P):.3f}(sd{st.pstdev(P):.3f}) "
          f"carry={st.mean(Cy):.3f}(sd{st.pstdev(Cy):.3f}) worst-comp={min(Cp):.0f}% shards~{int(st.mean(T))}")
json.dump(results,open("out_thickness_test/search/seeds2.json","w"),indent=1)
print("\n=== which configs RELIABLY pass the gate (>=5/6 seeds)? ===")
for k,v in results.items():
    tag="RELIABLE" if v["passed"]>=5 else f"unreliable ({v['passed']}/6)"
    print(f"  {k}: {tag} | primary {st.mean(v['primary']):.3f}±{st.pstdev(v['primary']):.3f} carry {st.mean(v['carry']):.3f}±{st.pstdev(v['carry']):.3f}")
