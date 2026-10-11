"""Gardner grading model: one pretrained image backbone with three heads.

  expansion        5 classes (Gardner grade 1-5)
  inner cell mass  3 classes (A/B/C), only learned for expansion grade >= 3
  trophectoderm    3 classes (A/B/C), only learned for expansion grade >= 3

Below expansion grade 3 the inner cell mass and trophectoderm aren't gradable,
so their prediction is reported as "not defined", same as the experts do.

The whole training set (about 150 MB as uint8) lives on the GPU and is
augmented there in batches, so there are no DataLoader worker processes.
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
from torchvision.transforms.v2 import functional as TF

from . import paths

os.environ.setdefault("HF_HOME", str(paths.CACHE / "huggingface"))  # keep model downloads off C:
import timm  # noqa: E402  (after HF_HOME)

CACHE_FILE = paths.PROCESSED / "images_u8_288x384.npy"
IMAGE_HW = (288, 384)
BACKGROUND = 136 / 255  # median border gray level, fills rotated-in corners
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
HEADS = {"exp": 5, "icm": 3, "te": 3}


class ImageBank:
    """Every cached image in memory (2,344 x 288 x 384 uint8 = 259 MB)."""

    def __init__(self):
        self.arr = np.load(CACHE_FILE)
        files = pd.read_csv(paths.PROCESSED / "images_index.csv")["file"]
        self.rows = {f: i for i, f in enumerate(files)}

    def get(self, files: list[str], device) -> torch.Tensor:
        """(B, 1, H, W) uint8 tensor."""
        idx = np.array([self.rows[f] for f in files])
        return torch.from_numpy(self.arr[idx]).unsqueeze(1).to(device)


def to_net(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) float in [0, 1] -> (B, 3, H, W) ImageNet-normalised."""
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x.expand(-1, 3, -1, -1) - mean) / std


def gpu_augment(x: torch.Tensor, background: float | None = BACKGROUND) -> torch.Tensor:
    """Random rotation (any angle), flip, zoom, shift, brightness, contrast and
    blur, drawn independently for every image in the batch (all channels of an
    image share the same geometry, so stacked video frames stay aligned).
    Embryos have no natural "up", so every rotation is a valid new example.

    x: (B, C, H, W) uint8 -> (B, C, H, W) float in [0, 1]
    background: fill for rotated-in corners; None uses each image's mean.
    """
    B, _, H, W = x.shape
    dev = x.device
    x = x.float() / 255
    fill = x.mean(dim=(1, 2, 3), keepdim=True) if background is None else background

    def U(lo, hi):
        return torch.empty(B, device=dev).uniform_(lo, hi)

    ang = U(0, 2 * math.pi)
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    zoom = U(0.87, 1.15)
    c, s = ang.cos() / zoom, ang.sin() / zoom
    # affine_grid works in [-1, 1] coordinates; rescale so rotation is in
    # pixel space and the 4:3 image doesn't get sheared.
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = c * flip
    theta[:, 0, 1] = -s * H / W
    theta[:, 1, 0] = s * flip * W / H
    theta[:, 1, 1] = c
    theta[:, 0, 2] = U(-0.08, 0.08)
    theta[:, 1, 2] = U(-0.08, 0.08)
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    inside = F.grid_sample(torch.ones_like(x), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    out = out + (1 - inside) * fill

    mean = out.mean(dim=(2, 3), keepdim=True)
    out = (out - mean) * U(0.75, 1.25).view(B, 1, 1, 1) + mean + U(-0.08, 0.08).view(B, 1, 1, 1)
    blur = torch.rand(B, device=dev) < 0.2
    if blur.any():
        sigma = float(np.random.uniform(0.3, 1.2))
        out[blur] = TF.gaussian_blur(out[blur], kernel_size=[5, 5], sigma=[sigma, sigma])
    return out.clamp(0, 1)


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
    """Square-root inverse-frequency weights, mean 1. Grade C is rare (16 of
    1,697 ICM labels), so without this the model would never predict it."""
    counts = np.bincount(labels[(labels >= 0) & (labels < n)], minlength=n).astype(float)
    w = 1 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def grade_loss(out: dict, y: torch.Tensor, w_icm: torch.Tensor, w_te: torch.Tensor, smoothing: float = 0.05):
    loss = F.cross_entropy(out["exp"].float(), y[:, 0], label_smoothing=smoothing)
    for k, col, w in (("icm", 1, w_icm), ("te", 2, w_te)):
        t = y[:, col]
        m = (t >= 0) & (t <= 2)  # 3 = not defined (expansion grade 1-2): no ICM/TE target
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


def train(files: list[str], targets: np.ndarray, cfg: TrainConfig, bank: ImageBank, log=print,
          monitor: tuple[list[str], np.ndarray] | None = None) -> nn.Module:
    """Train for a fixed number of epochs and return the EMA (weight-averaged)
    model. No early stopping, so held-out folds stay truly unseen."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    X = bank.get(files, dev)
    Y = torch.as_tensor(np.asarray(targets, dtype=np.int64), device=dev)
    steps = len(files) // cfg.batch
    model = Grader(cfg.arch).to(dev).to(memory_format=torch.channels_last)
    ema = timm.utils.ModelEmaV3(model, decay=cfg.ema_decay)
    w_icm = class_weights(np.asarray(targets)[:, 1], 3).to(dev)
    w_te = class_weights(np.asarray(targets)[:, 2], 3).to(dev)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.lr},
        {"params": model.heads.parameters(), "lr": cfg.lr * 10},
    ], weight_decay=cfg.weight_decay)
    total, warm = cfg.epochs * steps, steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / total))))
    scaler = torch.amp.GradScaler()
    for epoch in range(cfg.epochs):
        model.train()
        t0, running = time.time(), 0.0
        perm = torch.randperm(len(files), device=dev)
        for i in range(steps):
            idx = perm[i * cfg.batch:(i + 1) * cfg.batch]
            x = to_net(gpu_augment(X[idx])).contiguous(memory_format=torch.channels_last)
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


def _views(x: torch.Tensor) -> list[torch.Tensor]:
    """The 8 rotations/flips of a batch (test-time augmentation)."""
    out = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        out += [r, torch.flip(r, dims=(3,))]
    return out


@torch.no_grad()
def predict(model: nn.Module, files: list[str], bank: ImageBank, tta: bool = True,
            batch: int = 32) -> tuple[pd.DataFrame, np.ndarray]:
    """Class probabilities (averaged over 8 views) and pooled image features."""
    dev = next(model.parameters()).device
    model.eval()
    probs = {k: [] for k in HEADS}
    feats = []
    for i in range(0, len(files), batch):
        x = to_net(bank.get(files[i:i + batch], dev).float() / 255)
        views = _views(x) if tta else [x]
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
    """Turn probability columns into grade codes (ICM/TE = 3 'not defined'
    when the predicted expansion is grade 1-2)."""
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
