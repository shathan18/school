"""Clean test at <=300 TOTAL: does concentrating the budget on the face (dense face + very
coarse bg) beat a uniform 300-shard tiling on FACE quality? Calibrated to stay under cap."""
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
    kw=dict(face_masks=face_bin,face_density=fd,bg_coarsen=bgc) if fd>1 else {}
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack,seed=3,damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0,**kw)
    pred=r.render_color_np(C.stack_transmit_lut(names,sc,si)); tot=bs.get("A",{}).get("achieved",0)+bs.get("B",{}).get("achieved",0)
    fsA,feA=face_metrics(pred["A"],targets["A"],bbox["A"])
    return pred,tot,fsA,feA
# calibrate: coarse base + high bg_coarsen so total <= 300 even with dense face
cfgs=[("uniform ~290",0.112,1.0,1.0),
      ("face9 bg6",0.20,9.0,6.0),
      ("face16 bg8",0.22,16.0,8.0),
      ("face25 bg10",0.24,25.0,10.0),
      ("face40 bg12",0.26,40.0,12.0)]
print("Clean <=300 test: uniform vs face-concentrated")
res=[]
for lbl,fs,fd,bgc in cfgs:
    pred,tot,fsA,feA=run(fs,fd,bgc); res.append((lbl,pred,tot,fsA,feA))
    flag="" if tot<=300 else "  <-- OVER CAP (ignore)"
    print(f"  {lbl}: {tot} shards, faceSSIM_A={fsA:.3f} faceEdge_A={feA:.3f}{flag}")
y0,y1,x0,x1=bbox["A"]
fig,ax=plt.subplots(2,6,figsize=(20,7.5))
ax[0,0].imshow(np.clip(targets["A"],0,1),origin="lower",aspect="auto"); ax[0,0].set_title("target"); ax[0,0].axis("off")
ax[1,0].imshow(np.clip(targets["A"],0,1)[y0:y1,x0:x1],origin="lower",aspect="auto"); ax[1,0].axis("off")
for ci,(lbl,pred,tot,fsA,feA) in enumerate(res):
    ax[0,ci+1].imshow(np.clip(pred["A"],0,1),origin="lower",aspect="auto"); ax[0,ci+1].set_title(f"{lbl}\n{tot}sh fSSIM={fsA:.3f}",fontsize=9); ax[0,ci+1].axis("off")
    ax[1,ci+1].imshow(np.clip(pred["A"],0,1)[y0:y1,x0:x1],origin="lower",aspect="auto"); ax[1,ci+1].axis("off")
plt.suptitle("Face concentration at <=300 total: does dense-face beat uniform on the face?",fontsize=12)
plt.tight_layout(); plt.savefig("out_thickness_test/search/face_dual2.png",dpi=95,bbox_inches="tight"); print("saved face_dual2.png")
