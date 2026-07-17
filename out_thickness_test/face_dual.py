"""FOCUSED test: dual-density Voronoi (dense face, coarse elsewhere). Does the face resolve
at <=300 TOTAL shards by concentrating the budget on it? Sweep face_density + bg_coarsen."""
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
face={};bbox={}
for w in ("A","B"): face[w],bbox[w]=face_roi_from_target(targets[w])
face_bin={w:(face[w]>0.4) for w in ("A","B")}
panels,_=build_panels_greedy(scene,count=16,mode="deliberate",K=16,targets=targets,seed=3,angle_deg_range=(5,85))
def run(fragsize,fd,bgc):
    SP=dataclasses.replace(scene.solve,fragment_size=fragsize,fragment_min_area=scene.solve.fragment_min_area*(fragsize/0.135)**2,fragment_max_area=scene.solve.fragment_max_area*(fragsize/0.135)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=SP); table=build_projection_table(ts); r=Renderer(ts,table)
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack,seed=3,damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0,
        face_masks=face_bin,face_density=fd,bg_coarsen=bgc)
    pred=r.render_color_np(C.stack_transmit_lut(names,sc,si)); tot=bs.get("A",{}).get("achieved",0)+bs.get("B",{}).get("achieved",0)
    fsA,feA=face_metrics(pred["A"],targets["A"],bbox["A"])
    return pred,tot,fsA,feA
# tune base fragsize + face_density + bg_coarsen to keep total<=300 while concentrating on face
cfgs=[("control (uniform)",0.108,1.0,1.0),
      ("face 9x, bg 2x",0.12,9.0,2.0),
      ("face 16x, bg 3x",0.13,16.0,3.0),
      ("face 25x, bg 4x",0.14,25.0,4.0)]
print("Dual-density face Voronoi at <=300 total shards:")
res=[]
for lbl,fs,fd,bgc in cfgs:
    pred,tot,fsA,feA=run(fs,fd,bgc); res.append((lbl,pred,tot,fsA,feA))
    print(f"  {lbl}: {tot} shards total, faceSSIM_A={fsA:.3f} faceEdge_A={feA:.3f} {'(OVER CAP)' if tot>300 else ''}")
y0,y1,x0,x1=bbox["A"]
fig,ax=plt.subplots(2,5,figsize=(19,8))
ax[0,0].imshow(np.clip(targets["A"],0,1),origin="lower",aspect="auto"); ax[0,0].set_title("target"); ax[0,0].axis("off")
ax[1,0].imshow(np.clip(targets["A"],0,1)[y0:y1,x0:x1],origin="lower",aspect="auto"); ax[1,0].axis("off"); ax[1,0].set_title("face crop",fontsize=9)
for ci,(lbl,pred,tot,fsA,feA) in enumerate(res):
    ax[0,ci+1].imshow(np.clip(pred["A"],0,1),origin="lower",aspect="auto"); ax[0,ci+1].set_title(f"{lbl}\n{tot} sh, faceSSIM={fsA:.3f}",fontsize=9); ax[0,ci+1].axis("off")
    ax[1,ci+1].imshow(np.clip(pred["A"],0,1)[y0:y1,x0:x1],origin="lower",aspect="auto"); ax[1,ci+1].axis("off")
plt.suptitle("Dual-density: dense face + coarse background, <=300 total. Does the face resolve now?",fontsize=13)
plt.tight_layout(); plt.savefig("out_thickness_test/search/face_dual.png",dpi=95,bbox_inches="tight"); print("saved face_dual.png")
