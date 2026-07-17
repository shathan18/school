import sys, dataclasses, statistics as st
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
scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res; names=["clear"]+C.CMYK
GIRL=("examples/girl_front_nobg.png","examples/girl_back_nobg.png"); FOOD=("examples/apples.jpg","examples/breakfast.jpg")
def load(p): 
    t={"A":C.load_color_target(p[0],wr,white_thr=scene.white_threshold),"B":C.load_color_target(p[1],wr,white_thr=scene.white_threshold)}
    return t,{w:C.subject_mask(t[w],scene.white_threshold) for w in ("A","B")}
def build(pc,mult,K,seed,t):
    panels,_=build_panels_greedy(scene,count=pc,mode="deliberate",K=K,targets=t,seed=seed,angle_deg_range=(5,85))
    FS=0.135/np.sqrt(mult)
    SP=dataclasses.replace(scene.solve,fragment_size=FS,fragment_min_area=scene.solve.fragment_min_area*(FS/0.135)**2,fragment_max_area=scene.solve.fragment_max_area*(FS/0.135)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=SP); table=build_projection_table(ts); r=Renderer(ts,table)
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,t,names=names,white_thr=ts.white_threshold,max_stack=ts.color_max_stack,seed=seed,damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0)
    return ts,table,r,panels,C.stack_transmit_lut(names,sc,si),bs
def cmax(ts,table,r,panels,pT,s):
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}; best=None;bc=-1;c=[]
    for gi,p in enumerate(panels):
        w=prim[p.name]; pt=np.ones_like(pT); pt[gi]=pT[gi]; solo=r.render_color_np(pt)[w]
        cc=((1.0-solo.mean(-1)>0.05)&s[w]).sum()/s[w].sum(); c.append(cc)
    return max(c)
def busiest(ts,table,r,panels,pT,s,w):
    prim={p.name:primary_wall_of(ts,table,p) for p in panels}; best=None;bc=-1
    for gi,p in enumerate(panels):
        if prim[p.name]!=w: continue
        pt=np.ones_like(pT); pt[gi]=pT[gi]; solo=r.render_color_np(pt)[w]
        cc=((1.0-solo.mean(-1)>0.05)&s[w]).sum()/s[w].sum()
        if cc>bc: bc=cc;best=solo
    return best,bc
print("=== (a) WINNER 16/0.3/K16 on SECOND PAIR apples/breakfast ===")
t,s=load(FOOD); P=[];CP=[];T=[];Cy=[]
for seed in (1,2,3,4,5):
    ts,table,r,panels,pT,bs=build(16,0.3,16,seed,t); pred=r.render_color_np(pT); acc=_metrics.evaluate_wall_accuracy(t,pred)
    cA=100*(((1.0-pred["A"].mean(-1))>0.05)&s["A"]).sum()/s["A"].sum(); cB=100*(((1.0-pred["B"].mean(-1))>0.05)&s["B"]).sum()/s["B"].sum()
    prim=(acc["A"]["ssim"]+acc["B"]["ssim"]+acc["A"]["edge_fidelity"]+acc["B"]["edge_fidelity"])/2; car=cmax(ts,table,r,panels,pT,s)
    tot=bs.get("A",{}).get("achieved",0)+bs.get("B",{}).get("achieved",0); P.append(prim);CP.append(min(cA,cB));T.append(tot);Cy.append(car)
    print(f"  seed {seed}: primary={prim:.3f} carry={car:.3f} min-comp={min(cA,cB):.0f}% shards={tot} {'' if (min(cA,cB)>=85 and tot<=300) else '<-- FAILS'}")
print(f"  gate {sum(1 for c,x in zip(CP,T) if c>=85 and x<=300)}/5 | primary {st.mean(P):.3f}±{st.pstdev(P):.3f} carry {st.mean(Cy):.3f}±{st.pstdev(Cy):.3f}")
print("\n=== (b) render top-3 on girls ===")
t,s=load(GIRL)
configs=[("WINNER 16p/~65 shards/K16",16,0.3,16,3),("runner-up 20p/~103/K32",20,0.5,32,3),("CURRENT 8p/~192/K16",8,1.0,16,3)]
fig,ax=plt.subplots(3,4,figsize=(16,13))
for ri,(lbl,pc,mult,K,seed) in enumerate(configs):
    ts,table,r,panels,pT,bs=build(pc,mult,K,seed,t); full=r.render_color_np(pT)
    ax[ri,0].imshow(np.clip(full["A"],0,1),aspect="auto"); ax[ri,0].set_ylabel(lbl,fontsize=10)
    ax[ri,1].imshow(np.clip(full["B"],0,1),aspect="auto")
    sA,cA=busiest(ts,table,r,panels,pT,s,"A"); sB,cB=busiest(ts,table,r,panels,pT,s,"B")
    ax[ri,2].imshow(np.clip(sA,0,1),aspect="auto"); ax[ri,2].set_title(f"busiest A alone: {cA*100:.0f}%",fontsize=9)
    ax[ri,3].imshow(np.clip(sB,0,1),aspect="auto"); ax[ri,3].set_title(f"busiest B alone: {cB*100:.0f}%",fontsize=9)
    if ri==0: ax[0,0].set_title("Wall A (front)",fontweight="bold"); ax[0,1].set_title("Wall B (back)",fontweight="bold")
    for c in range(4): ax[ri,c].axis("off")
plt.suptitle("Background-removed girls, top-3: shadows (1-2) + one panel alone (3-4)",fontsize=12)
plt.tight_layout(); plt.savefig("out_thickness_test/search/girl_top3.png",dpi=90,bbox_inches="tight"); print("saved girl_top3.png")
# also a shard-level look at the winner so user can judge crispness vs count
fig,ax=plt.subplots(2,5,figsize=(19,8))
for ri,w in enumerate(("A","B")): ax[ri,0].imshow(np.clip(t[w],0,1),aspect="auto"); ax[ri,0].axis("off"); ax[ri,0].set_title("target" if ri==0 else "")
for ci,mult in enumerate((0.3,0.5,1.0,1.8)):
    ts,table,r,panels,pT,bs=build(16,mult,16,3,t); pred=r.render_color_np(pT); tot=bs.get("A",{}).get("achieved",0)+bs.get("B",{}).get("achieved",0)
    for ri,w in enumerate(("A","B")): ax[ri,ci+1].imshow(np.clip(pred[w],0,1),aspect="auto"); ax[ri,ci+1].axis("off")
    ax[0,ci+1].set_title(f"{tot} shards",fontweight="bold")
plt.suptitle("Winner config (16 panels) at different shard counts — how crisp do you want the girl?",fontsize=12)
plt.tight_layout(); plt.savefig("out_thickness_test/search/girl_shardlevels.png",dpi=90,bbox_inches="tight"); print("saved girl_shardlevels.png")
