"""Multi-seed validation on Vermeers: candidates from fine sweep, 6 seeds each.
Report gate reliability, primary mean+-sd, carry mean+-sd."""
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
A_IMG="examples/WhatsApp Image 2026-07-16 at 11.39.33.jpeg"; B_IMG="examples/WhatsApp Image 2026-07-16 at 11.39.33 (1).jpeg"
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
    cA=100*(((1.0-pred["A"].mean(-1))>0.05)&subject["A"]).sum()/subject["A"].sum()
    cB=100*(((1.0-pred["B"].mean(-1))>0.05)&subject["B"]).sum()/subject["B"].sum()
    prim=(acc["A"]["ssim"]+acc["B"]["ssim"]+acc["A"]["edge_fidelity"]+acc["B"]["edge_fidelity"])/2
    return prim,cryptic(ts,table,r,panels,pT),min(cA,cB),nA+nB
CANDS=[(12,0.5),(16,0.5),(20,0.5),(16,0.3)]
SEEDS=[1,2,3,4,5,6]; R={}
for pc,mult in CANDS:
    P=[];Cy=[];CP=[];T=[]
    print(f"\n--- panels={pc}, mult={mult} ---")
    for s in SEEDS:
        prim,carry,mc,tot=evaluate(pc,mult,s); P.append(prim);Cy.append(carry);CP.append(mc);T.append(tot)
        print(f"  seed {s}: primary={prim:.3f} carry={carry:.3f} min-comp={mc:.0f}% shards={tot} {'' if (mc>=85 and tot<=300) else '<-- FAILS'}")
    passed=sum(1 for c,t in zip(CP,T) if c>=85 and t<=300)
    R[f"{pc}_{mult}"]=dict(passed=passed,primary=P,carry=Cy,comp=CP,total=T)
    print(f"  gate {passed}/6 | primary {st.mean(P):.3f}±{st.pstdev(P):.3f} carry {st.mean(Cy):.3f}±{st.pstdev(Cy):.3f} worst-comp {min(CP):.0f}% shards~{int(st.mean(T))}")
json.dump(R,open("out_thickness_test/search/vermeer_seeds.json","w"),indent=1)
print("\n=== reliable (>=5/6)? ===")
for k,v in R.items():
    print(f"  {k}: {'RELIABLE' if v['passed']>=5 else 'unreliable'} ({v['passed']}/6) | primary {st.mean(v['primary']):.3f}±{st.pstdev(v['primary']):.3f} carry {st.mean(v['carry']):.3f}±{st.pstdev(v['carry']):.3f}")
