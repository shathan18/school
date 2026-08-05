import sys, os, shutil, win32com.client
root = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(root, "_deckcheck")
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
app = win32com.client.Dispatch("PowerPoint.Application")
pres = app.Presentations.Open(os.path.join(root, "MoP26_ShadowArt_final.pptx"),
                              WithWindow=False)
for i, sl in enumerate(pres.Slides, 1):
    sl.Export(os.path.join(out, f"s{i:02d}.png"), "PNG", 1600, 900)
pres.Close(); app.Quit()
print("exported", len(os.listdir(out)))
