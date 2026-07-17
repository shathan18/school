"""Final validation: (a) winner 16/mult0.7/K32 on the SECOND pair apples/breakfast (5 seeds);
(b) fair comparison of winner vs the CURRENT-CONFIG baseline (8 panels, K16) on flat portraits."""
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
scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res
names=["clear"]+C.CMYK
FLAT=("examples/Screenshot 2026-07-16 102515.png","examples/Screenshot 2026-07-16 102536.png")
FOOD=("examples/apples.jpg","examples/breakfast.jpg")
def load(pair):
    t={"A":C.load_color_target(pair[0],wr,white_thr=scene.white_threshold),
       "B":C.load_color_target(pair[1],wr,white_thr=scene.white_threshold)}
    s={w:C.subject_mask(t[w],scene.white_threshold) for w in ("A","B")}
    return t,s
def cryptic(ts,table,r,panels,pT,subject):
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}; carry=[]
    for gi,p in enumerate(panels):
        w=prim[p.name]; pt=np.ones_like(pT); pt[gi]=pT[gi]
        d=(1.0-r.render_color_np(pt)[w].mean(-1))>0.05
        carry.append((d&subject[w]).sum()/max(subject[w].sum(),1))
    return max(carry)
def evaluate(pc,mult,K,seed,targets,subject):
    panels,_=build_panels_greedy(scene,count=pc,mode="deliberate",K=K,targets=targets,seed=seed,angle_deg_range=(5,85))
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
    return prim,cryptic(ts,table,r,panels,pT,subject),min(cA,cB),nA+nB
def multiseed(label,pc,mult,K,pair,seeds):
    t,s=load(pair); P=[];Cy=[];CP=[];T=[]
    print(f"\n--- {label} ---")
    for sd in seeds:
        prim,carry,mc,tot=evaluate(pc,mult,K,sd,t,s); P.append(prim);Cy.append(carry);CP.append(mc);T.append(tot)
        print(f"  seed {sd}: primary={prim:.3f} carry={carry:.3f} min-comp={mc:.0f}% shards={tot} {'' if (mc>=85 and tot<=300) else '<-- FAILS'}")
    passed=sum(1 for c,t2 in zip(CP,T) if c>=85 and t2<=300)
    print(f"  gate {passed}/{len(seeds)} | primary {st.mean(P):.3f}±{st.pstdev(P):.3f} carry {st.mean(Cy):.3f}±{st.pstdev(Cy):.3f} worst-comp {min(CP):.0f}%")
    return dict(passed=passed,n=len(seeds),primary=P,carry=Cy,comp=CP,total=T)

R={}
print("=== (a) WINNER on SECOND PAIR (apples/breakfast) ===")
R["winner_food"]=multiseed("WINNER 16/0.7/K32 on apples/breakfast",16,0.7,32,FOOD,[1,2,3,4,5])
print("\n=== (b) does the WINNER beat the CURRENT CONFIG? both on flat portraits ===")
R["winner_flat"]=multiseed("WINNER 16/0.7/K32 on flat portraits",16,0.7,32,FLAT,[1,2,3,4,5,6])
R["baseline_flat"]=multiseed("BASELINE 8/1.0/K16 on flat portraits (current config)",8,1.0,16,FLAT,[1,2,3,4,5,6])
json.dump(R,open("out_thickness_test/search/validate.json","w"),indent=1)
print("\nsaved validate.json")
