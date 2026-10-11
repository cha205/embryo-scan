"""Download and unpack the Kromp et al. 2023 blastocyst dataset (624 MB, CC BY 4.0).

Source: https://doi.org/10.6084/m9.figshare.20123153.v3

Usage: python scripts/blastocyst/download_data.py
"""

import argparse
import hashlib
import shutil
import urllib.request
import zipfile

from embryoscan import paths

URL = "https://ndownloader.figshare.com/files/39348899"
MD5 = "d19532b4b6bc4792b44738b8930d9ad2"


def md5sum(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    paths.KROMP.mkdir(parents=True, exist_ok=True)
    if not paths.KROMP_ZIP.exists():
        print(f"downloading {URL} -> {paths.KROMP_ZIP}")
        tmp = paths.KROMP_ZIP.with_suffix(".part")
        with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.rename(paths.KROMP_ZIP)
    digest = md5sum(paths.KROMP_ZIP)
    if digest != MD5:
        raise SystemExit(f"checksum mismatch: {digest} != {MD5}; delete the zip and retry")
    print("checksum ok")
    if not paths.KROMP_IMAGES.exists():
        with zipfile.ZipFile(paths.KROMP_ZIP) as z:
            z.extractall(paths.KROMP)
    print(f"{len(list(paths.KROMP_IMAGES.glob('*.png')))} images in {paths.KROMP_IMAGES}")


if __name__ == "__main__":
    main()
