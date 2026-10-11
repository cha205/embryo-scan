"""Turn the 311k time-lapse frames into one compact array for training.

For each video: find the embryo (the most textured region inside the well,
averaged over a dozen frames), crop 288x288 around it, shrink to 160x160, and
write every frame into data/processed/timelapse_u8_160.npy. Reading one big
file sequentially is far faster than 311k small JPEGs on a hard drive.

Also writes data/processed/timelapse_frames.csv (one row per frame) and the
patient-level video split splits/timelapse_split.csv.

Usage: python scripts/prepare_timelapse.py [--workers 4]
"""

import argparse
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from embryoscan import paths
from embryoscan.timelapse import load_frames_table, make_video_split

OUT = paths.PROCESSED / "timelapse_u8_160.npy"
SIZE, CROP = 160, 288


def frame_files(video: str) -> dict[int, str]:
    root = paths.GOMEZ / "embryo_dataset" / video
    out = {}
    for p in root.rglob("*.jpeg"):
        m = re.search(r"RUN(\d+)", p.name)
        if m:
            out[int(m.group(1))] = str(p)
    return out


def embryo_center(files: list[str]) -> tuple[int, int]:
    """Centre of the most textured region (the embryo), averaged over frames.
    The well's bright rim is excluded by ignoring a border and very bright pixels."""
    acc = None
    for f in files:
        g = Image.open(f).convert("L")
        a = np.asarray(g, dtype=np.float32)
        edges = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        edges[a > 200] = 0
        acc = edges if acc is None else acc + edges
    sm = np.asarray(Image.fromarray(acc / acc.max() * 255).convert("L").filter(ImageFilter.GaussianBlur(25)), dtype=np.float32)
    h, w = sm.shape
    m = 90
    sm[:m], sm[-m:], sm[:, :m], sm[:, -m:] = 0, 0, 0, 0
    y, x = np.unravel_index(sm.argmax(), sm.shape)
    return int(x), int(y)


def process(job):
    video, rows, frames = job
    files = frame_files(video)
    sample = [files[f] for f in frames[:: max(1, len(frames) // 12)] if f in files][:12]
    cx, cy = embryo_center(sample) if sample else (250, 250)
    half = CROP // 2
    x0, y0 = min(max(cx - half, 0), 500 - CROP), min(max(cy - half, 0), 500 - CROP)
    arr = np.load(OUT, mmap_mode="r+")
    missing = []
    for r, f in zip(rows, frames):
        if f not in files:
            missing.append(r)
            continue
        g = Image.open(files[f]).convert("L").crop((x0, y0, x0 + CROP, y0 + CROP))
        arr[r] = np.asarray(g.resize((SIZE, SIZE), Image.Resampling.BILINEAR, reducing_gap=2.0))
    arr.flush()
    return video, (x0, y0), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    df = load_frames_table()
    np.lib.format.open_memmap(OUT, mode="w+", dtype=np.uint8, shape=(len(df), SIZE, SIZE)).flush()
    jobs = [(v, g.index.to_numpy(), g["frame"].to_numpy()) for v, g in df.groupby("video", sort=False)]
    crops, missing = {}, []
    with Pool(a.workers) as pool:
        for i, (video, xy, miss) in enumerate(pool.imap_unordered(process, jobs, chunksize=2)):
            crops[video] = xy
            missing += miss
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} videos", flush=True)
    df["missing"] = False
    df.loc[missing, "missing"] = True
    df["crop_x"] = df.video.map(lambda v: crops[v][0])
    df["crop_y"] = df.video.map(lambda v: crops[v][1])
    df.to_csv(paths.PROCESSED / "timelapse_frames.csv", index_label="row")
    split = make_video_split(df["video"].drop_duplicates())
    split.to_csv(paths.SPLITS / "timelapse_split.csv", index=False)
    print(f"{len(df)} frames, {len(missing)} missing files")
    print(split.groupby("split").agg(videos=("video", "count"), patients=("patient", "nunique")))


if __name__ == "__main__":
    main()
