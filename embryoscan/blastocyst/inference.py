"""Grade a single embryo photo with the 5-model ensemble, plus Grad-CAM heat maps.

Used by the demo server (server/main.py) and scripts/blastocyst/export_demo.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.ndimage import gaussian_filter

from .. import paths
from ..common.augment import dihedral_views, to_net
from .data import gardner_label
from .grading import DEFAULT_RUN, HEADS, IMAGE_HW, Grader, decode


def load_image(img: Image.Image) -> torch.Tensor:
    """Any embryo photo -> (1, 1, 288, 384) uint8, matching the training cache.

    Photos that aren't 4:3 are centre-cropped to 4:3 first instead of being
    stretched; the dataset's own 512 x 384 photos are unchanged. Phone photos
    are turned upright from their EXIF orientation."""
    h, w = IMAGE_HW
    g = ImageOps.exif_transpose(img).convert("L")
    scale = min(g.width / w, g.height / h)
    cw, ch = round(w * scale), round(h * scale)
    left, top = (g.width - cw) // 2, (g.height - ch) // 2
    g = g.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(g).copy())[None, None]


def load_ensemble(run: str = DEFAULT_RUN, device: str | None = None, folds=range(5)) -> list[nn.Module]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for k in folds:
        m = Grader(pretrained=False)
        m.load_state_dict(torch.load(paths.run_dir("blastocyst", run) / f"fold-{k}.pt", map_location=device))
        models.append(m.to(device).eval())
    return models


@torch.no_grad()
def grade(models: list[nn.Module], gray_u8: torch.Tensor, tta: bool = True) -> dict:
    """Average class probabilities over models and the 8 rotations/flips."""
    dev = next(models[0].parameters()).device
    x = to_net(gray_u8.to(dev).float() / 255)
    views = dihedral_views(x) if tta else [x]
    acc = {k: 0 for k in HEADS}
    for m in models:
        for v in views:
            out = m(v)
            for k in HEADS:
                acc[k] = acc[k] + out[k].float().softmax(-1)[0].cpu().numpy()
    n = len(models) * len(views)
    probs = {k: (acc[k] / n).tolist() for k in HEADS}
    row = {f"p_{k}{c}": probs[k][c] for k in HEADS for c in range(len(probs[k]))}
    d = decode(pd.DataFrame([row])).iloc[0]
    return {
        "grade": gardner_label(d["exp"], d["icm"], d["te"]),
        "codes": {"exp": int(d["exp"]), "icm": int(d["icm"]), "te": int(d["te"])},
        "probs": probs,
    }


class _Head(nn.Module):
    def __init__(self, model: nn.Module, head: str):
        super().__init__()
        self.model, self.head = model, head

    def forward(self, x):
        return self.model(x)[self.head]


def _undo(cam: np.ndarray, k: int, flip: bool) -> np.ndarray:
    if flip:
        cam = np.flip(cam, axis=-1)
    return np.rot90(cam, -k, axes=(-2, -1))


def heatmap(model: nn.Module, gray_u8: torch.Tensor, head: str, cls: int, smooth: float = 9.0) -> np.ndarray:
    """Grad-CAM for one head and class, averaged over the 8 rotations/flips.
    Grad-CAM is computed on the network's 9x12 feature grid, so it's smoothed
    (Gaussian, `smooth` px) to remove the blocky upsampling.
    Returns (288, 384) floats in [0, 1]."""
    dev = next(model.parameters()).device
    x = to_net(gray_u8.to(dev).float() / 255)
    cam = GradCAM(model=_Head(model, head), target_layers=[model.backbone.stages[-1]])
    acc = 0
    for k in range(4):
        for flip in (False, True):
            v = torch.rot90(x, k, dims=(2, 3))
            if flip:
                v = torch.flip(v, dims=(3,))
            acc = acc + _undo(cam(input_tensor=v, targets=[ClassifierOutputTarget(cls)])[0], k, flip)
    acc = gaussian_filter(acc / 8, smooth) if smooth else acc / 8
    return (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)
