"""Export sample embryos for the web demo: the photo, the ensemble's grade, and
Grad-CAM heat maps (where the model looked) for each grading head.

Picks gold-test images (never seen by any grading model) whose expert-panel
grade is one of WANTED, preferring images the experts agreed on most.

Usage: python scripts/export_demo_samples.py [--run convnext_t]
"""

import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image

from embryoscan import paths
from embryoscan.inference import grade, heatmap, load_ensemble, load_image

OUT = paths.ROOT / "web" / "public" / "samples"
WANTED = [("4AA", (3, 0, 0)), ("4BB", (3, 1, 1)), ("3BC", (2, 1, 2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="convnext_t")
    a = ap.parse_args()

    grades = pd.read_csv(paths.PROCESSED / "grades.csv")
    gold = grades[(grades.label_set == "gold") & grades[["exp", "icm", "te"]].notna().all(axis=1)]
    models = load_ensemble(a.run)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label, (e, i, t) in WANTED:
        pick = gold[(gold.exp == e) & (gold.icm == i) & (gold.te == t)].sort_values(
            ["exp_agreement", "icm_agreement", "te_agreement"], ascending=False)
        if pick.empty:
            print(f"no gold image graded {label}")
            continue
        f = pick.iloc[0]["file"]
        img = Image.open(paths.IMAGES / f)
        x = load_image(img)
        res = grade(models, x)
        sid = f.removesuffix(".png")
        (OUT / sid).mkdir(exist_ok=True)
        img.convert("L").save(OUT / sid / "image.jpg", quality=90)
        cams = {}
        for head, code in res["codes"].items():
            cls = code if head == "exp" or code < 3 else 0
            cam = heatmap(models[0], x, head, cls)
            Image.fromarray((cam * 255).astype(np.uint8)).resize((192, 144), Image.Resampling.BILINEAR).save(OUT / sid / f"cam_{head}.png")
            cams[head] = f"samples/{sid}/cam_{head}.png"
        manifest.append({"id": sid, "image": f"samples/{sid}/image.jpg", "cams": cams, "expert": label,
                         "model": res["grade"], "probs": {h: [round(p, 3) for p in v] for h, v in res["probs"].items()}})
        print(f"{sid}: expert panel {label}, model {res['grade']}")
    (OUT / "samples.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
