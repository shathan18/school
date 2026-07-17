"""Second-pair confirmation: does the semantic-null + face-saturation finding hold on a
DIFFERENT oil painting pair (Mona Lisa / real Pearl)? Control vs semantic, 4 seeds."""
import sys, dataclasses, statistics as st
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school\out_thickness_test")
import numpy as np
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics
from semantic_lib import face_roi_from_target, face_metrics
scene=load_scene("scenes/example.yaml"); wr=scene.solve.wall_res; names=["clear"]+C.CMYK
targets={"A":C.load_color_target("examples/Mona_Lisa.jpg",wr,white_thr=scene.white_threshold),
         "B":C.load_color_target("examples/The_Girl_With_The_Pearl_Earring.jpg",wr,white_thr=scene.white_threshold)}
sem={};bbox={}
for w in ("A","B"): sem[w],bbox[w]=face_roi_from_target(targets[w])
def evaluate(sw,seed):
    panels,_=build_panels_greedy(scene,count=16,mode="deliberate",K=16,targets=targets,seed=seed,angle_deg_range=(5,85))
    FS=0.135/np.sqrt(0.5)
    SP=dataclasses.replace(scene.solve,fragment_size=FS,fragment_min_area=scene.solve.fragment_min_area*(FS/0.135)**2,fragment_max_area=scene.solve.fragment_max_area*(FS/0.135)**2)
    ts=dataclasses.replace(scene,panels=panels,solve=SP); table=build_projection_table(ts); r=Renderer(ts,table)
    sc,op,fr,rs,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,white_thr=ts.white_threshold,
        max_stack=ts.color_max_stack,seed=seed,damage_weight=0.5,credit_weight=0.5,agree_min=0.90,shard_budget=0,semantic_masks=sem,semantic_weight=sw)
    pred=r.render_color_np(C.stack_transmit_lut(names,sc,si))
    fsA,feA=face_metrics(pred["A"],targets["A"],bbox["A"]); fsB,feB=face_metrics(pred["B"],targets["B"],bbox["B"])
    return (fsA+fsB)/2
print("SECOND PAIR (Mona Lisa / real Pearl), 16 panels mult0.5. Does semantic help the face?")
for sw in (0.0,0.5,1.0):
    vals=[evaluate(sw,s) for s in (1,2,3,4)]
    print(f"  sw={sw}: faceSSIM {st.mean(vals):.3f}±{st.pstdev(vals):.3f}")
