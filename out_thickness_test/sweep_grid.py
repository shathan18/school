"""Angle-diversity x shard-density sweep with subject-aware good/bad cross-talk metrics.
Targets: Wall B bad-crosstalk < 15%, and RMSE/SSIM within ~10% of best (A 0.224/0.758, B 0.253/0.732)."""
import sys, dataclasses, json
sys.path.insert(0, r"c:\Users\User1\Downloads\matterOfPerspective\school")
import numpy as np
from shadowart.config.io import load_scene
from shadowart.geometry.projection import build_projection_table, primary_wall_of
from shadowart.forward.renderer import Renderer
from shadowart.solve import decompose
from shadowart.solve.panel_search import build_panels_greedy
from shadowart.targets import color as C
from shadowart import metrics as _metrics

scene = load_scene("scenes/example.yaml"); wr = scene.solve.wall_res
names = ["clear"] + C.CMYK
targets = {"A": C.load_color_target("examples/apples.jpg", wr, white_thr=scene.white_threshold),
           "B": C.load_color_target("examples/breakfast.jpg", wr, white_thr=scene.white_threshold)}
subject = {w: C.subject_mask(targets[w], scene.white_threshold) for w in ("A","B")}

ANGLE = {"near-axis": dict(angle_bands=[(0,20),(70,90)]),
         "moderate":  dict(angle_deg_range=(20,70)),
         "wide":      dict(angle_deg_range=(5,85))}
DENSITY = {"0.5x": 0.190, "1x": 0.135, "2x": 0.095}
SEEDS = [3, 5]

def crosstalk_split(ts, table, renderer, panel_T, panels):
    """Return per-wall dict: total dark, good%, bad% (subject-aware)."""
    prim = {p.name: primary_wall_of(ts, table, p) for p in panels}
    def render_keep(keep):
        pt = panel_T.copy()
        for gi,p in enumerate(panels):
            if p.name not in keep: pt[gi]=1.0
        return renderer.render_color_np(pt)
    dark = lambda img: (1.0-img.mean(axis=-1))>0.05
    out={}
    for w in ("A","B"):
        nonprim = {p.name for p in panels if prim[p.name]!=w}
        full_d = dark(render_keep({p.name for p in panels})[w]); tot=full_d.sum()
        xt_d = dark(render_keep(nonprim)[w]) if nonprim else np.zeros_like(full_d)
        bad = (xt_d & ~subject[w]).sum(); good=(xt_d & subject[w]).sum()
        out[w]={"bad_pct":100*bad/max(tot,1),"good_pct":100*good/max(tot,1)}
    return out

results=[]
for aname, akw in ANGLE.items():
    for seed in SEEDS:
        panels,_ = build_panels_greedy(scene, count=14, mode="deliberate", K=16,
                                       targets=targets, seed=seed, **akw)
        a_ct = sum(1 for p in panels if primary_wall_of(dataclasses.replace(scene,panels=panels),
                   build_projection_table(dataclasses.replace(scene,panels=panels)),p)=="A")
        for dname, fs in DENSITY.items():
            sp = dataclasses.replace(scene.solve, fragment_size=fs,
                 fragment_min_area=scene.solve.fragment_min_area*(fs/scene.solve.fragment_size)**2,
                 fragment_max_area=scene.solve.fragment_max_area*(fs/scene.solve.fragment_size)**2)
            ts = dataclasses.replace(scene, panels=panels, solve=sp)
            table=build_projection_table(ts); renderer=Renderer(ts,table)
            sc,op,fr,res,sd,bs,si=decompose.fragment_shards_overlap(ts,table,targets,names=names,
                white_thr=ts.white_threshold,max_stack=ts.color_max_stack,seed=seed)
            panel_T=C.stack_transmit_lut(names,sc,si)
            pred=renderer.render_color_np(panel_T)
            acc=_metrics.evaluate_wall_accuracy(targets,pred)
            ct=crosstalk_split(ts,table,renderer,panel_T,panels)
            nA=bs.get("A",{}).get("achieved",0); nB=bs.get("B",{}).get("achieved",0)
            r={"angle":aname,"density":dname,"seed":seed,"split_A":a_ct,"split_B":14-a_ct,
               "shards_A":nA,"shards_B":nB,
               "A_rmse":acc["A"]["rmse"],"A_ssim":acc["A"]["ssim"],"A_edge":acc["A"]["edge_fidelity"],
               "B_rmse":acc["B"]["rmse"],"B_ssim":acc["B"]["ssim"],"B_edge":acc["B"]["edge_fidelity"],
               "A_bad":ct["A"]["bad_pct"],"A_good":ct["A"]["good_pct"],
               "B_bad":ct["B"]["bad_pct"],"B_good":ct["B"]["good_pct"]}
            results.append(r)
            print(f"{aname:9s} {dname:4s} seed{seed}: split {r['split_A']}/{r['split_B']} "
                  f"shards {nA}/{nB} | B_bad={r['B_bad']:.1f}% B_good={r['B_good']:.1f}% "
                  f"| A {acc['A']['rmse']:.3f}/{acc['A']['ssim']:.3f} B {acc['B']['rmse']:.3f}/{acc['B']['ssim']:.3f}")

with open("out_thickness_test/sweep_grid_results.json","w") as f:
    json.dump(results,f,indent=1)
print(f"\nwrote {len(results)} configs to out_thickness_test/sweep_grid_results.json")
