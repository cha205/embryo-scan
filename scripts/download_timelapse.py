"""Download the Gomez et al. 2022 human embryo time-lapse dataset (in-focus plane
only, 12.2 GB) plus its annotations, frame timestamps and blastocyst grades.

Source: https://doi.org/10.5281/zenodo.7912264 (CC BY-NC-SA 4.0)
Zenodo throttles each connection (~1.3 MB/s), so large files are fetched as
64 MB byte ranges over several connections. Interrupted downloads resume,
MD5 is checked, and archives are extracted safely.

Usage: python scripts/download_timelapse.py [--delete-archives] [--connections 6]
"""

import argparse
import hashlib
import json
import math
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from embryoscan import paths

CHUNK = 64 << 20

URL = "https://zenodo.org/api/records/7912264/files/{}/content"
FILES = {  # name: (md5, bytes)
    "embryo_dataset_annotations.tar.gz": ("c9b6e725a6c032b3f94d838580ac2e53", 53267),
    "embryo_dataset_time_elapsed.tar.gz": ("01bd49e679cb19ec67172bbab5495fdc", 859452),
    "embryo_dataset_grades.csv": ("25eca34656454d3cfc6d8c59305b200e", 9810),
    "embryo_dataset.tar.gz": ("f8f7c72e40725331dd16d6d89b8e6c32", 12147455757),
}


def md5sum(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, size: int, connections: int) -> None:
    """Fetch the file as CHUNK-sized byte ranges in parallel into a
    preallocated .part file. A sidecar JSON records finished chunks so an
    interrupted run picks up where it stopped."""
    dest = paths.GOMEZ / name
    if dest.exists():
        return
    part = dest.with_name(dest.name + ".part")
    state = dest.with_name(dest.name + ".chunks.json")
    n = math.ceil(size / CHUNK)
    if state.exists():
        done = set(json.loads(state.read_text()))
    else:  # a plain sequential .part from an older run covers its first chunks
        have = part.stat().st_size if part.exists() else 0
        done = {i for i in range(n) if min(size, (i + 1) * CHUNK) <= have}
    with open(part, "r+b" if part.exists() else "w+b") as f:
        f.truncate(size)
    lock = threading.Lock()
    progress = {"bytes": sum(min(size, (i + 1) * CHUNK) - i * CHUNK for i in done), "t0": time.time(), "last": 0.0}
    start_bytes = progress["bytes"]

    def fetch(i: int) -> None:
        lo, hi = i * CHUNK, min(size, (i + 1) * CHUNK) - 1
        for attempt in range(6):
            got = 0
            try:
                req = urllib.request.Request(URL.format(name), headers={"Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(req, timeout=60) as r, open(part, "r+b") as f:
                    if r.status != 206:
                        raise OSError("server ignored the byte range")
                    f.seek(lo)
                    while piece := r.read(1 << 18):
                        f.write(piece)
                        got += len(piece)
                if got != hi - lo + 1:
                    raise OSError(f"short read {got}")
                with lock:
                    done.add(i)
                    state.write_text(json.dumps(sorted(done)))
                    progress["bytes"] += got
                    now = time.time()
                    if now - progress["last"] > 30:
                        rate = (progress["bytes"] - start_bytes) / (now - progress["t0"]) / 1e6
                        eta = (size - progress["bytes"]) / max(rate * 1e6, 1) / 60
                        print(f"  {progress['bytes'] / 1e9:.2f} / {size / 1e9:.2f} GB, {rate:.1f} MB/s, ~{eta:.0f} min left", flush=True)
                        progress["last"] = now
                return
            except (OSError, MemoryError) as e:  # MemoryError: the machine ran short of RAM; wait and retry
                print(f"  chunk {i} attempt {attempt + 1} failed: {type(e).__name__} {e}", flush=True)
                time.sleep(3 * (attempt + 1))
        raise SystemExit(f"chunk {i} kept failing; rerun to resume")

    todo = [i for i in range(n) if i not in done]
    with ThreadPoolExecutor(connections) as ex:
        list(ex.map(fetch, todo))
    state.unlink()
    part.rename(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-archives", action="store_true", help="delete each .tar.gz after extracting it")
    ap.add_argument("--connections", type=int, default=6)
    a = ap.parse_args()
    paths.GOMEZ.mkdir(parents=True, exist_ok=True)
    for name, (md5, size) in FILES.items():
        marker = paths.GOMEZ / f".{name}.extracted"
        if marker.exists():
            print(f"{name}: already extracted")
            continue
        print(f"{name} ({size / 1e6:,.1f} MB)", flush=True)
        download(name, size, a.connections)
        dest = paths.GOMEZ / name
        if dest.stat().st_size != size or md5sum(dest) != md5:
            raise SystemExit(f"checksum mismatch for {name}; delete it and retry")
        print("  checksum ok", flush=True)
        if name.endswith(".tar.gz"):
            with tarfile.open(dest) as tar:
                tar.extractall(paths.GOMEZ, filter="data")  # blocks absolute paths and ../ tricks
            marker.touch()
            print("  extracted", flush=True)
            if a.delete_archives:
                dest.unlink()
    print("done")


if __name__ == "__main__":
    main()
