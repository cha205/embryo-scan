"""Turn the 311k time-lapse frames into one compact array for training.

For each video: find the embryo (the most textured region inside the well,
averaged over a dozen frames), crop 288x288 around it, shrink to 160x160, and
write every frame into data/processed/timelapse_u8_160.npy. Reading one big
file sequentially is far faster than 311k small JPEGs on a hard drive.

On a hard drive, *how* the JPEGs are read matters most: parallel readers make
the disk seek constantly (~100 files/s). One reader going through the files
in creation order (the order they were unpacked, so roughly their order on
disk) manages ~1,000 files/s, while a thread pool decodes JPEGs in parallel.

Also writes data/processed/timelapse_frames.csv (one row per frame) and the
patient-level video split splits/timelapse_split.csv.

Usage: python scripts/prepare_timelapse.py [--threads 5]
"""

import argparse
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageFilter

from embryoscan import paths
from embryoscan.timelapse import load_frames_table, make_video_split

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few source JPEGs end early; keep the part that's there
OUT = paths.PROCESSED / "timelapse_u8_160.npy"
SIZE, CROP = 160, 288


def video_files(video: str) -> list[tuple[float, int, str]]:
    """(creation time, frame number, path) for every JPEG of a video."""
    out, stack = [], [paths.GOMEZ / "embryo_dataset" / video]
    while stack:
        for e in os.scandir(stack.pop()):
            if e.is_dir():
                stack.append(e.path)
            elif (m := re.search(r"RUN(\d+)\.jpe?g$", e.name)):
                st = e.stat()
                out.append((getattr(st, "st_birthtime", st.st_ctime), int(m.group(1)), e.path))
    return out


def decode(data: bytes) -> np.ndarray | None:
    try:
        return np.asarray(Image.open(BytesIO(data)).convert("L"))
    except OSError:
        return None


def embryo_center(images: list[np.ndarray]) -> tuple[int, int]:
    """Centre of the most textured region (the embryo), averaged over frames.
    The well's bright rim is excluded by ignoring a border and very bright pixels."""
    acc = None
    for a in images:
        edges = np.asarray(Image.fromarray(a).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        edges[a > 200] = 0
        acc = edges if acc is None else acc + edges
    sm = np.asarray(Image.fromarray(acc / acc.max() * 255).convert("L").filter(ImageFilter.GaussianBlur(25)), dtype=np.float32)
    m = 90
    sm[:m], sm[-m:], sm[:, :m], sm[:, -m:] = 0, 0, 0, 0
    y, x = np.unravel_index(sm.argmax(), sm.shape)
    return int(x), int(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args()
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    df = load_frames_table()
    groups = {v: g for v, g in df.groupby("video", sort=False)}
    listing = {v: video_files(v) for v in groups}
    order = sorted(groups, key=lambda v: min((t for t, _, _ in listing[v]), default=0))  # unpack order
    arr = np.lib.format.open_memmap(OUT, mode="w+", dtype=np.uint8, shape=(len(df), SIZE, SIZE))

    # reader thread: whole videos, files in creation order, a couple of videos ahead
    q: queue.Queue = queue.Queue(maxsize=3)

    def reader():
        for v in order:
            data = {}
            for _, frame, path in sorted(listing[v]):
                with open(path, "rb") as f:
                    data[frame] = f.read()
            q.put((v, data))
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    crops, missing = {}, []
    t0, done = time.time(), 0
    with ThreadPoolExecutor(a.threads) as pool:
        while (item := q.get()) is not None:
            video, data = item
            g = groups[video]
            rows, frames = g.index.to_numpy(), g["frame"].to_numpy()
            present = [f for f in frames if f in data]
            sample = [data[f] for f in present[:: max(1, len(present) // 12)]][:12]
            imgs = [im for im in pool.map(decode, sample) if im is not None]
            cx, cy = embryo_center(imgs) if imgs else (250, 250)
            half = CROP // 2
            x0, y0 = min(max(cx - half, 0), 500 - CROP), min(max(cy - half, 0), 500 - CROP)
            crops[video] = (x0, y0)

            def work(rf):
                r, f = rf
                im = decode(data[f]) if f in data else None
                if im is None:
                    return r, None
                crop = Image.fromarray(im[y0:y0 + CROP, x0:x0 + CROP])
                return r, np.asarray(crop.resize((SIZE, SIZE), Image.Resampling.BILINEAR, reducing_gap=2.0))

            for r, out in pool.map(work, zip(rows, frames)):
                if out is None:
                    missing.append(r)
                else:
                    arr[r] = out
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(order)} videos ({time.time() - t0:.0f}s)", flush=True)
    arr.flush()
    df["missing"] = False
    df.loc[missing, "missing"] = True
    df["crop_x"] = df.video.map(lambda v: crops[v][0])
    df["crop_y"] = df.video.map(lambda v: crops[v][1])
    df.to_csv(paths.PROCESSED / "timelapse_frames.csv", index_label="row")
    split = make_video_split(df["video"].drop_duplicates())
    split.to_csv(paths.SPLITS / "timelapse_split.csv", index=False)
    print(f"{len(df)} frames, {len(missing)} missing or unreadable, {time.time() - t0:.0f}s")
    print(split.groupby("split").agg(videos=("video", "count"), patients=("patient", "nunique")))


if __name__ == "__main__":
    main()
