"""Gardner grading model: a pretrained backbone with three heads.

The heads predict expansion (grade 1-5), inner cell mass (A-C) and
trophectoderm (A-C). Like the experts, the model only grades inner cell mass
and trophectoderm from expansion grade 3 up.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import paths
from ..common.augment import dihedral_views, gpu_augment, to_net

os.environ.setdefault("HF_HOME", str(paths.CACHE / "huggingface"))  # must be set before timm is imported
import timm  # noqa: E402

DEFAULT_RUN = "grader-convnext-tiny"
IMAGE_HW = (288, 384)
BACKGROUND = 136 / 255  # median border gray of the photos, fills rotated-in corners
HEADS = {"exp": 5, "icm": 3, "te": 3}


class ImageBank:
    """All cached images in memory (2,344 x 288 x 384 uint8, 259 MB)."""

    def __init__(self):
        self.arr = np.load(paths.BLASTOCYST_IMAGES)
        files = pd.read_csv(paths.BLASTOCYST_IMAGE_INDEX)["file"]
        self.rows = {f: i for i, f in enumerate(files)}

    def get(self, files: list[str], device) -> torch.Tensor:
        """(B, 1, H, W) uint8 tensor."""
        idx = np.array([self.rows[f] for f in files])
        return torch.from_numpy(self.arr[idx]).unsqueeze(1).to(device)


class Grader(nn.Module):
    def __init__(self, arch: str = "convnext_tiny.fb_in22k_ft_in1k", pretrained: bool = True, drop: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.drop = nn.Dropout(drop)
        self.heads = nn.ModuleDict({k: nn.Linear(d, n) for k, n in HEADS.items()})

    def forward(self, x, return_features: bool = False):
        f = self.backbone(x)
        out = {k: h(self.drop(f)) for k, h in self.heads.items()}
        return (out, f) if return_features else out


def class_weights(labels: np.ndarray, n: int) -> torch.Tensor:
    """Square-root inverse-frequency weights with mean 1. Grade C is rare
    (16 of 1,697 ICM labels) and is otherwise never predicted."""
    counts = np.bincount(labels[(labels >= 0) & (labels < n)], minlength=n).astype(float)
    w = 1 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def grade_loss(out: dict, y: torch.Tensor, w_icm: torch.Tensor, w_te: torch.Tensor, smoothing: float = 0.05):
    loss = F.cross_entropy(out["exp"].float(), y[:, 0], label_smoothing=smoothing)
    for k, col, w in (("icm", 1, w_icm), ("te", 2, w_te)):
        t = y[:, col]
        m = (t >= 0) & (t <= 2)  # 3 means not graded
        if m.any():
            loss = loss + F.cross_entropy(out[k][m].float(), t[m], weight=w, label_smoothing=smoothing)
    return loss


@dataclass
class TrainConfig:
    arch: str = "convnext_tiny.fb_in22k_ft_in1k"
    epochs: int = 25
    batch: int = 16
    lr: float = 1e-4  # backbone; heads use 10x
    weight_decay: float = 0.05
    ema_decay: float = 0.998
    seed: int = 0


def train(
    files: list[str],
    targets: np.ndarray,
    cfg: TrainConfig,
    bank: ImageBank,
    log=print,
    monitor: tuple[list[str], np.ndarray] | None = None,
) -> nn.Module:
    """Train for a fixed number of epochs and return the EMA model. There's no
    early stopping, so the held-out fold never influences training."""
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    X = bank.get(files, dev)
    Y = torch.as_tensor(np.asarray(targets, dtype=np.int64), device=dev)
    steps = len(files) // cfg.batch
    model = Grader(cfg.arch).to(dev).to(memory_format=torch.channels_last)
    ema = timm.utils.ModelEmaV3(model, decay=cfg.ema_decay)
    w_icm = class_weights(np.asarray(targets)[:, 1], 3).to(dev)
    w_te = class_weights(np.asarray(targets)[:, 2], 3).to(dev)
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg.lr},
            {"params": model.heads.parameters(), "lr": cfg.lr * 10},
        ],
        weight_decay=cfg.weight_decay,
    )
    total, warm = cfg.epochs * steps, steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / total)))
    )
    scaler = torch.amp.GradScaler()
    for epoch in range(cfg.epochs):
        model.train()
        t0, running = time.time(), 0.0
        perm = torch.randperm(len(files), device=dev)
        for i in range(steps):
            idx = perm[i * cfg.batch : (i + 1) * cfg.batch]
            x = to_net(gpu_augment(X[idx], BACKGROUND)).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            loss = grade_loss(out, Y[idx], w_icm, w_te)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            running += loss.item()
        msg = f"epoch {epoch + 1:2d}/{cfg.epochs} loss {running / steps:.3f} ({time.time() - t0:.0f}s)"
        if monitor is not None and ((epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1):
            acc = quick_accuracy(ema.module, *monitor, bank=bank)
            msg += " | out-of-fold acc " + " ".join(f"{k} {v:.3f}" for k, v in acc.items())
        log(msg)
    del X
    return ema.module.eval()


@torch.no_grad()
def predict(
    model: nn.Module, files: list[str], bank: ImageBank, tta: bool = True, batch: int = 32
) -> tuple[pd.DataFrame, np.ndarray]:
    """Class probabilities and pooled image features, averaged over the 8
    rotations/flips when tta is on."""
    dev = next(model.parameters()).device
    model.eval()
    probs = {k: [] for k in HEADS}
    feats = []
    for i in range(0, len(files), batch):
        x = to_net(bank.get(files[i : i + batch], dev).float() / 255)
        views = dihedral_views(x) if tta else [x]
        acc = {k: 0 for k in HEADS}
        facc = 0
        for v in views:
            with torch.autocast("cuda", dtype=torch.float16):
                out, f = model(v.contiguous(memory_format=torch.channels_last), return_features=True)
            for k in HEADS:
                acc[k] = acc[k] + out[k].float().softmax(-1)
            facc = facc + f.float()
        for k in HEADS:
            probs[k].append((acc[k] / len(views)).cpu().numpy())
        feats.append((facc / len(views)).cpu().numpy())
    cols = {}
    for k, n in HEADS.items():
        p = np.concatenate(probs[k])
        for c in range(n):
            cols[f"p_{k}{c}"] = p[:, c]
    return decode(pd.DataFrame({"file": files, **cols})), np.concatenate(feats)


def decode(df: pd.DataFrame) -> pd.DataFrame:
    """Probability columns to grade codes. ICM and TE get code 3 (not graded)
    when the predicted expansion is grade 1-2."""
    df = df.copy()
    df["exp"] = df[[f"p_exp{c}" for c in range(5)]].to_numpy().argmax(1)
    early = df["exp"] <= 1
    for k in ("icm", "te"):
        df[k] = df[[f"p_{k}{c}" for c in range(3)]].to_numpy().argmax(1)
        df.loc[early, k] = 3
    return df


def quick_accuracy(model: nn.Module, files: list[str], targets: np.ndarray, bank: ImageBank) -> dict:
    pred, _ = predict(model, files, bank, tta=False)
    t = np.asarray(targets)
    out = {"exp": float((pred["exp"].to_numpy() == t[:, 0]).mean())}
    m = (t[:, 0] >= 2) & (pred["exp"].to_numpy() >= 2)
    for k, col in (("icm", 1), ("te", 2)):
        out[k] = float((pred[k].to_numpy()[m] == t[m, col]).mean()) if m.any() else float("nan")
    return out
