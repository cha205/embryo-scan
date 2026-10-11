"""Frame-by-frame development-stage model for time-lapse videos.

Input for frame t is a 3-channel stack of frames t-3, t and t+3 (about 45
minutes apart), so the network sees motion: a cell that is mid-division looks
different from one that just sits there.
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

from . import paths
from .grading import gpu_augment, to_net
from .timelapse import PHASES

os.environ.setdefault("HF_HOME", str(paths.CACHE / "huggingface"))
import timm  # noqa: E402

CACHE = paths.PROCESSED / "timelapse_u8_160.npy"
OFFSET = 3  # frames between the stacked channels


def load_table() -> pd.DataFrame:
    df = pd.read_csv(paths.PROCESSED / "timelapse_frames.csv")
    split = pd.read_csv(paths.SPLITS / "timelapse_split.csv")
    return df.merge(split[["video", "split"]], on="video")


_NEIGHBOURS = None


def neighbours(offset: int = OFFSET) -> tuple[np.ndarray, np.ndarray]:
    """For every cache row, the row `offset` frames earlier/later in the same
    video (clamped at the ends). Rows of a video are contiguous and in time order."""
    global _NEIGHBOURS
    if _NEIGHBOURS is None:
        df = pd.read_csv(paths.PROCESSED / "timelapse_frames.csv", usecols=["row", "video"])
        pos = df.groupby("video").cumcount().to_numpy()
        size = df.groupby("video")["video"].transform("size").to_numpy()
        rows = df["row"].to_numpy()
        prev, nxt = np.empty(len(df), np.int64), np.empty(len(df), np.int64)
        prev[rows] = rows - np.minimum(pos, offset)
        nxt[rows] = rows + np.minimum(size - 1 - pos, offset)
        _NEIGHBOURS = (prev, nxt)
    return _NEIGHBOURS


def stack(arr: np.ndarray, rows: np.ndarray, prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    return np.stack([arr[prev], arr[rows], arr[nxt]], axis=1)


def make_model(arch: str = "resnet34.a1_in1k", pretrained: bool = True) -> nn.Module:
    return timm.create_model(arch, pretrained=pretrained, num_classes=len(PHASES), drop_rate=0.2)


@dataclass
class StageConfig:
    arch: str = "resnet34.a1_in1k"
    epochs: int = 8
    batch: int = 96
    lr: float = 6e-4
    stride: int = 3  # train on every 3rd frame
    seed: int = 0


def train(df: pd.DataFrame, cfg: StageConfig, log=print, val: pd.DataFrame | None = None) -> nn.Module:
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    arr = np.load(CACHE, mmap_mode="r")
    prev, nxt = neighbours()
    keep = ((df.groupby("video").cumcount() % cfg.stride) == 0).to_numpy() & (df["phase"].to_numpy() >= 0) & ~df["missing"].to_numpy()
    rows = np.sort(df["row"].to_numpy()[keep])  # file order: fast reads from a hard drive
    log(f"loading {len(rows)} training stacks to the GPU")
    t0 = time.time()
    X = torch.empty((len(rows), 3, arr.shape[1], arr.shape[2]), dtype=torch.uint8, device=dev)
    for s in range(0, len(rows), 4096):
        r = rows[s:s + 4096]
        X[s:s + len(r)] = torch.from_numpy(stack(arr, r, prev[r], nxt[r])).to(dev)
    phase_by_row = dict(zip(df["row"], df["phase"]))
    labels = np.array([phase_by_row[r] for r in rows])
    Y = torch.as_tensor(labels, device=dev)
    log(f"loaded in {time.time() - t0:.0f}s")

    counts = np.bincount(labels, minlength=len(PHASES)).astype(float)
    w = 1 / np.sqrt(np.maximum(counts, 1))
    weights = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    model = make_model(cfg.arch).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
    steps = len(rows) // cfg.batch
    total, warm = cfg.epochs * steps, steps // 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / total))))
    scaler = torch.amp.GradScaler()
    for epoch in range(cfg.epochs):
        model.train()
        t0, running = time.time(), 0.0
        perm = torch.randperm(len(rows), device=dev)
        for i in range(steps):
            idx = perm[i * cfg.batch:(i + 1) * cfg.batch]
            x = to_net(gpu_augment(X[idx], background=None)).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            loss = F.cross_entropy(out.float(), Y[idx], weight=weights, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        msg = f"epoch {epoch + 1}/{cfg.epochs} loss {running / steps:.3f} ({time.time() - t0:.0f}s)"
        if val is not None:
            v = val.iloc[::5]
            p = predict_frames(model, v["row"].to_numpy())
            ok = v["phase"].to_numpy()
            m = ok >= 0
            msg += f" | val frame acc {(p.argmax(1)[m] == ok[m]).mean():.3f}"
        log(msg)
    del X
    torch.cuda.empty_cache()
    return model.eval()


@torch.no_grad()
def predict_frames(model: nn.Module, rows: np.ndarray, batch: int = 256, tta: bool = False) -> np.ndarray:
    """Phase log-probabilities for the given cache rows. With tta, averages the
    4 right-angle rotations."""
    dev = next(model.parameters()).device
    model.eval()
    arr = np.load(CACHE, mmap_mode="r")
    prev, nxt = neighbours()
    out = []
    for s in range(0, len(rows), batch):
        r = rows[s:s + batch]
        x = to_net(torch.from_numpy(stack(arr, r, prev[r], nxt[r])).to(dev).float() / 255)
        views = [torch.rot90(x, k, dims=(2, 3)) for k in range(4)] if tta else [x]
        acc = 0
        for v in views:
            with torch.autocast("cuda", dtype=torch.float16):
                acc = acc + model(v.contiguous(memory_format=torch.channels_last)).float().softmax(-1)
        out.append((acc / len(views)).clamp_min(1e-6).log().cpu().numpy())
    return np.concatenate(out)
