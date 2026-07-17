"""ShadowArt — computational shadow-art fabrication.

Take two target images + a scene description of a room corner (two walls at 90 deg,
two floor point-lights, a woven lattice of transparent panels), compute what to make
opaque on each panel so that BOTH wall images appear when lit, preview the predicted
result, and export machine-ready cut files (DXF + SVG).

Pipeline (see cli.py):
    images + scene  ->  geometry (homographies)  ->  forward renderer
                    ->  solve (joint 2-wall optimization)  ->  raster2vec
                    ->  fabricate (joints, nesting, export)

v1 target: two sharp monochrome images. Color (CMYK, 5 layers) is a designed-in,
deferred extension (see targets/color.py, fabricate/layers.py).
"""

__version__ = "0.1.0"
