"""At ~2750 shards: is the face resolution from SHARD COUNT (sw=0 control) or the SEMANTIC MASK?"""
import sys, dataclasses
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school\out_thickness_test")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from semantic_lib import face_roi_from_target, face_metrics
scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res; names=["clear"]+C.CMYK
targets={"A":C.load_color_target("examples/girl_front_nobg.png",wr,white_thr=scene.white_threshold),
         "B":C.load_color_target("examples/girl_back_nobg.png",wr,white_thr=scene.white_threshold)}
sem={};bbox={}
for w in ("A","B"): sem[w],bbox[w]=face_roi_from_target(targets[w])
panels,_=build_panels_greedy(scene,count=16,mode="deliberate",K=16,targets=targets,seed=3,angle_deg_range=(5,85))
def run(sw):
    fs=0.034
    SP=dataclasses.replace(scene.solve,fragment_size=fs,fragment_min_area=scene.solve.fragment_min_area*(fs/0.135)**2,fragment_max_area=scene.solve.fragment_max_area*(fs/0.135)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=SP); table=build_projection_table(ts); r=Renderer(ts,table)
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack,seed=3,damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0,
        semantic_masks=sem,semantic_weight=sw)
    pred=r.render_color_np(C.stack_transmit_lut(names,sc,si)); tot=bs.get("A",{}).get("achieved",0)+bs.get("B",{}).get("achieved",0)
    fsA,feA=face_metrics(pred["A"],targets["A"],bbox["A"])
    print(f"  sw={sw}: {tot} shards, faceSSIM_A={fsA:.3f} faceEdge_A={feA:.3f}")
    return pred,tot,fsA,feA
print("~2750 shards: control (semantic OFF) vs semantic ON")
p0,t0,fs0,fe0=run(0.0); p1,t1,fs1,fe1=run(1.0)
print(f"\n  delta from semantic mask: faceSSIM {fs1-fs0:+.3f}, faceEdge {fe1-fe0:+.3f}")
print(f"  -> {'SEMANTIC MASK matters at high counts' if abs(fs1-fs0)>0.03 else 'it is the SHARD COUNT, not the mask (semantic ~null even here)'}")
y0,y1,x0,x1=bbox["A"]
fig,ax=plt.subplots(2,3,figsize=(13,8))
imgs=[("target",np.clip(targets["A"],0,1),None),(f"control sw=0\n{t0} sh, faceSSIM={fs0:.3f}",np.clip(p0["A"],0,1),None),(f"semantic sw=1\n{t1} sh, faceSSIM={fs1:.3f}",np.clip(p1["A"],0,1),None)]
for ci,(lbl,im,_) in enumerate(imgs):
    ax[0,ci].imshow(im,origin="lower",aspect="auto"); ax[0,ci].set_title(lbl,fontsize=10); ax[0,ci].axis("off")
    ax[1,ci].imshow(im[y0:y1,x0:x1],origin="lower",aspect="auto"); ax[1,ci].axis("off")
plt.suptitle("~2750 shards: does the semantic mask matter, or is it just shard count?",fontsize=12)
plt.tight_layout(); plt.savefig("out_thickness_test/search/highshards_control.png",dpi=95,bbox_inches="tight"); print("saved highshards_control.png")
