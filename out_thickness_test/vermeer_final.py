"""Vermeer final: (a) second-pair validation of winner 20/0.5/K32 on apples/breakfast (5 seeds);
(b) render top-3 Vermeer configs (walls + busiest solo panel)."""
import sys, dataclasses, json, statistics as st
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res
names=["clear"]+C.CMYK
VERM=("examples/WhatsApp Image 2026-07-16 at 11.39.33.jpeg","examples/WhatsApp Image 2026-07-16 at 11.39.33 (1).jpeg")
FOOD=("examples/apples.jpg","examples/breakfast.jpg")
def load(pair):
    t={"A":C.load_color_target(pair[0],wr,white_thr=scene.white_threshold),"B":C.load_color_target(pair[1],wr,white_thr=scene.white_threshold)}
    return t,{w:C.subject_mask(t[w],scene.white_threshold) for w in ("A","B")}
def build(pc,mult,K,seed,targets):
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
    return ts,table,r,panels,C.stack_transmit_lut(names,sc,si),bs
def carrymax(ts,table,r,panels,pT,subject):
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}; c=[]
    for gi,p in enumerate(panels):
        w=prim[p.name]; pt=np.ones_like(pT); pt[gi]=pT[gi]
        d=(1.0-r.render_color_np(pt)[w].mean(-1))>0.05; c.append((d&subject[w]).sum()/max(subject[w].sum(),1))
    return max(c)

# (a) second pair
print("=== (a) WINNER 20/0.5/K32 on SECOND PAIR (apples/breakfast) ===")
t,s=load(FOOD)
P=[];Cy=[];CP=[];T=[]
for seed in (1,2,3,4,5):
    ts,table,r,panels,pT,bs=build(20,0.5,32,seed,t); pred=r.render_color_np(pT)
    acc=_metrics.evaluate_wall_accuracy(t,pred)
    cA=100*(((1.0-pred["A"].mean(-1))>0.05)&s["A"]).sum()/s["A"].sum(); cB=100*(((1.0-pred["B"].mean(-1))>0.05)&s["B"]).sum()/s["B"].sum()
    prim=(acc["A"]["ssim"]+acc["B"]["ssim"]+acc["A"]["edge_fidelity"]+acc["B"]["edge_fidelity"])/2
    car=carrymax(ts,table,r,panels,pT,s); tot=bs.get("A",{}).get("achieved",0)+bs.get("B",{}).get("achieved",0)
    P.append(prim);Cy.append(car);CP.append(min(cA,cB));T.append(tot)
    print(f"  seed {seed}: primary={prim:.3f} carry={car:.3f} min-comp={min(cA,cB):.0f}% shards={tot} {'' if (min(cA,cB)>=85 and tot<=300) else '<-- FAILS'}")
print(f"  gate {sum(1 for c,x in zip(CP,T) if c>=85 and x<=300)}/5 | primary {st.mean(P):.3f}±{st.pstdev(P):.3f} carry {st.mean(Cy):.3f}±{st.pstdev(Cy):.3f}")

# (b) render top-3 on Vermeers
print("\n=== (b) rendering top-3 Vermeer configs ===")
t,s=load(VERM)
def busiest(ts,table,r,panels,pT,w):
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}; best=None;bc=-1
    for gi,p in enumerate(panels):
        if prim[p.name]!=w: continue
        pt=np.ones_like(pT); pt[gi]=pT[gi]; solo=r.render_color_np(pt)[w]
        c=((1.0-solo.mean(-1)>0.05)&s[w]).sum()/s[w].sum()
        if c>bc: bc=c;best=solo
    return best,bc
configs=[("WINNER 20p/162 shards/K32",20,0.5,32,4),("runner-up 20p/162/K48",20,0.5,48,4),("CURRENT 8p/~224/K16",8,0.7,16,3)]
fig,ax=plt.subplots(3,4,figsize=(17,13))
for ri,(lbl,pc,mult,K,seed) in enumerate(configs):
    ts,table,r,panels,pT,bs=build(pc,mult,K,seed,t); full=r.render_color_np(pT)
    ax[ri,0].imshow(np.clip(full["A"],0,1),origin="lower",aspect="auto"); ax[ri,0].set_ylabel(lbl,fontsize=10)
    ax[ri,1].imshow(np.clip(full["B"],0,1),origin="lower",aspect="auto")
    sA,cA=busiest(ts,table,r,panels,pT,"A"); sB,cB=busiest(ts,table,r,panels,pT,"B")
    ax[ri,2].imshow(np.clip(sA,0,1),origin="lower",aspect="auto"); ax[ri,2].set_title(f"busiest A alone: {cA*100:.0f}%",fontsize=9)
    ax[ri,3].imshow(np.clip(sB,0,1),origin="lower",aspect="auto"); ax[ri,3].set_title(f"busiest B alone: {cB*100:.0f}%",fontsize=9)
    if ri==0: ax[0,0].set_title("Wall A (front)",fontweight="bold"); ax[0,1].set_title("Wall B (back)",fontweight="bold")
    for c in range(4): ax[ri,c].set_xticks([]);ax[ri,c].set_yticks([])
plt.suptitle("Vermeer top-3: shadows (1-2) + one panel alone (3-4). Real photographic paintings: soft but cryptic.",fontsize=12)
plt.tight_layout(); plt.savefig("out_thickness_test/search/vermeer_top3.png",dpi=88,bbox_inches="tight")
print("saved vermeer_top3.png")
