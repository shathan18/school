"""Does the oil-painting face RESOLVE with far more shards (well past the 300 cap)?
300 -> 3000 shards, semantic mask ON (face-focused) = best case. Render face crops."""
import sys, dataclasses, time
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
from shadowart import metrics as _metrics
from semantic_lib import face_roi_from_target, face_metrics
scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res; names=["clear"]+C.CMYK
targets={"A":C.load_color_target("examples/girl_front_nobg.png",wr,white_thr=scene.white_threshold),
         "B":C.load_color_target("examples/girl_back_nobg.png",wr,white_thr=scene.white_threshold)}
sem={};bbox={}
for w in ("A","B"): sem[w],bbox[w]=face_roi_from_target(targets[w])
panels,_=build_panels_greedy(scene,count=16,mode="deliberate",K=16,targets=targets,seed=3,angle_deg_range=(5,85))
def run(fragsize,sw):
    SP=dataclasses.replace(scene.solve,fragment_size=fragsize,
       fragment_min_area=scene.solve.fragment_min_area*(fragsize/0.135)**2,
       fragment_max_area=scene.solve.fragment_max_area*(fragsize/0.135)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=SP); table=build_projection_table(ts); r=Renderer(ts,table)
    t0=time.time()
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack,seed=3,damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0,
        semantic_masks=sem,semantic_weight=sw)
    pred=r.render_color_np(C.stack_transmit_lut(names,sc,si))
    tot=bs.get("A",{}).get("achieved",0)+bs.get("B",{}).get("achieved",0)
    fsA,feA=face_metrics(pred["A"],targets["A"],bbox["A"])
    print(f"  fragsize={fragsize:.3f} sw={sw}: {tot} shards, faceSSIM_A={fsA:.3f} faceEdge_A={feA:.3f}  ({time.time()-t0:.0f}s)")
    return pred,tot,fsA
# target ~300,1000,2000,3000 shards -> fragment_size = 0.135*sqrt(192/N)
levels=[(0.108,"~300"),(0.059,"~1000"),(0.042,"~2000"),(0.034,"~3000")]
print("Face resolution vs shard count (semantic ON, face-focused):")
res=[]
for fs,lbl in levels:
    pred,tot,fsA=run(fs,1.0); res.append((lbl,pred,tot,fsA))
y0,y1,x0,x1=bbox["A"]
fig,ax=plt.subplots(2,5,figsize=(19,8))
ax[0,0].imshow(np.clip(targets["A"],0,1),origin="lower",aspect="auto"); ax[0,0].set_title("target"); ax[0,0].axis("off")
ax[1,0].imshow(np.clip(targets["A"],0,1)[y0:y1,x0:x1],origin="lower",aspect="auto"); ax[1,0].axis("off"); ax[1,0].set_title("face crop",fontsize=9)
for ci,(lbl,pred,tot,fsA) in enumerate(res):
    ax[0,ci+1].imshow(np.clip(pred["A"],0,1),origin="lower",aspect="auto"); ax[0,ci+1].set_title(f"{tot} shards\nfaceSSIM={fsA:.3f}",fontsize=10); ax[0,ci+1].axis("off")
    ax[1,ci+1].imshow(np.clip(pred["A"],0,1)[y0:y1,x0:x1],origin="lower",aspect="auto"); ax[1,ci+1].axis("off")
plt.suptitle("Does the face resolve with MORE shards? 300 -> 3000 (far past the 300 fabrication cap)",fontsize=13)
plt.tight_layout(); plt.savefig("out_thickness_test/search/highshards.png",dpi=95,bbox_inches="tight"); print("saved highshards.png")
