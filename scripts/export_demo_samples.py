"""Export sample embryos for the web demo: the photo, the model's grade, and
Grad-CAM heat maps (where the model looked) for each grading head.

Grad-CAM is averaged over the 8 rotations/flips of the image, which makes the
maps steadier than a single pass.

Usage: python scripts/export_demo_samples.py --run convnext_t --fold 0
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from embryoscan import paths
from embryoscan.data import gardner_label
from embryoscan.grading import HEADS, Grader, ImageBank, predict, to_net

OUT = paths.ROOT / "web" / "public" / "samples"
WANTED = [("4AA", (3, 0, 0)), ("4BB", (3, 1, 1)), ("3BC", (2, 1, 2))]


class Head(nn.Module):
    def __init__(self, model, head):
        super().__init__()
        self.model, self.head = model, head

    def forward(self, x):
        return self.model(x)[self.head]


def views(x):
    for k in range(4):
        for flip in (False, True):
            v = torch.rot90(x, k, dims=(2, 3))
            yield (torch.flip(v, dims=(3,)) if flip else v), k, flip


def undo(cam, k, flip):
    if flip:
        cam = np.flip(cam, axis=-1)
    return np.rot90(cam, -k, axes=(-2, -1))


def tta_cam(model, x, head, cls):
    cam = GradCAM(model=Head(model, head), target_layers=[model.backbone.stages[-1]])
    acc = 0
    for v, k, flip in views(x):
        acc = acc + undo(cam(input_tensor=v, targets=[ClassifierOutputTarget(cls)])[0], k, flip)
    acc = acc / 8
    return (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="convnext_t")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()

    grades = pd.read_csv(paths.PROCESSED / "grades.csv")
    gold = grades[grades.label_set == "gold"].copy()
    for c in ("exp", "icm", "te"):
        gold = gold[gold[c].notna()]
    model = Grader(pretrained=False).cuda().eval()
    model.load_state_dict(torch.load(paths.RUNS / "grader" / a.run / f"fold{a.fold}.pt", map_location="cuda"))
    bank = ImageBank()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label, (e, i, t) in WANTED:
        pick = gold[(gold.exp == e) & (gold.icm == i) & (gold.te == t)].sort_values(
            ["exp_agreement", "icm_agreement", "te_agreement"], ascending=False)
        if pick.empty:
            print(f"no gold image graded {label}")
            continue
        f = pick.iloc[0]["file"]
        pred, _ = predict(model, [f], bank)
        p = pred.iloc[0]
        x = to_net(bank.get([f], "cuda").float() / 255)
        sid = f.replace(".png", "")
        (OUT / sid).mkdir(exist_ok=True)
        Image.open(paths.IMAGES / f).convert("L").save(OUT / sid / "image.jpg", quality=90)
        cams = {}
        for head in HEADS:
            cls = int(p[head]) if head == "exp" or int(p[head]) < 3 else 0
            cam = tta_cam(model, x, head, cls)
            Image.fromarray((cam * 255).astype(np.uint8)).resize((192, 144), Image.Resampling.BILINEAR).save(OUT / sid / f"cam_{head}.png")
            cams[head] = f"samples/{sid}/cam_{head}.png"
        manifest.append({
            "id": sid, "image": f"samples/{sid}/image.jpg", "cams": cams,
            "expert": label, "model": gardner_label(p["exp"], p["icm"], p["te"]),
            "probs": {h: [round(float(p[f"p_{h}{c}"]), 3) for c in range(n)] for h, n in HEADS.items()},
        })
        print(f"{sid}: expert {label}, model {manifest[-1]['model']}")
    (OUT / "samples.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
