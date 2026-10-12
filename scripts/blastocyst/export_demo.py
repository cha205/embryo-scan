"""Export sample embryos for the Analyze tab: photo, ensemble grade and Grad-CAM heat maps.

Uses gold-test images (never seen by a grading model) with the panel grades in
WANTED, preferring the ones the experts agreed on most. Writes
web/public/blastocyst/samples.json and one folder per sample.

Usage: python scripts/blastocyst/export_demo.py [--run grader-convnext-tiny]
"""

import argparse
import json
import shutil

import numpy as np
import pandas as pd
from PIL import Image

from embryoscan import paths
from embryoscan.blastocyst.grading import DEFAULT_RUN
from embryoscan.blastocyst.inference import grade, heatmap, load_ensemble, load_image

OUT = paths.WEB_PUBLIC / "blastocyst"
WANTED = [("4AA", (3, 0, 0)), ("4BB", (3, 1, 1)), ("3BC", (2, 1, 2))]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=DEFAULT_RUN, help="grader run folder under runs/blastocyst/")
    a = ap.parse_args()

    grades = pd.read_csv(paths.BLASTOCYST_GRADES)
    gold = grades[(grades.label_set == "gold") & grades[["exp", "icm", "te"]].notna().all(axis=1)]
    models = load_ensemble(a.run)
    shutil.rmtree(OUT / "samples", ignore_errors=True)  # drop samples from earlier exports
    manifest = []
    for label, (e, i, t) in WANTED:
        pick = gold[(gold.exp == e) & (gold.icm == i) & (gold.te == t)].sort_values(
            ["exp_agreement", "icm_agreement", "te_agreement"], ascending=False
        )
        if pick.empty:
            print(f"no gold image graded {label}")
            continue
        f = pick.iloc[0]["file"]
        img = Image.open(paths.KROMP_IMAGES / f)
        x = load_image(img)
        res = grade(models, x)
        sid = f.removesuffix(".png")
        folder = OUT / "samples" / sid
        folder.mkdir(parents=True, exist_ok=True)
        img.convert("L").save(folder / "photo.jpg", quality=90)
        heatmaps = {}
        for head, code in res["codes"].items():
            cls = code if head == "exp" or code < 3 else 0
            cam = heatmap(models[0], x, head, cls)
            Image.fromarray((cam * 255).astype(np.uint8)).resize((192, 144), Image.Resampling.BILINEAR).save(
                folder / f"heatmap-{head}.png"
            )
            heatmaps[head] = f"/blastocyst/samples/{sid}/heatmap-{head}.png"
        manifest.append(
            {
                "id": sid,
                "photo": f"/blastocyst/samples/{sid}/photo.jpg",
                "heatmaps": heatmaps,
                "expert": label,
                "model": res["grade"],
                "probs": {h: [round(p, 3) for p in v] for h, v in res["probs"].items()},
            }
        )
        print(f"{sid}: expert panel {label}, model {res['grade']}")
    (OUT / "samples.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
