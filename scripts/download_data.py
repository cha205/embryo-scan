"""Download and unpack the Kromp et al. 2023 blastocyst dataset (624 MB, CC BY 4.0).

Source: https://doi.org/10.6084/m9.figshare.20123153.v3
Usage:  python scripts/download_data.py
"""

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
    paths.RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not paths.RAW_ZIP.exists():
        print(f"downloading {URL} -> {paths.RAW_ZIP}")
        tmp = paths.RAW_ZIP.with_suffix(".part")
        with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.rename(paths.RAW_ZIP)
    digest = md5sum(paths.RAW_ZIP)
    if digest != MD5:
        raise SystemExit(f"checksum mismatch: {digest} != {MD5}; delete the zip and retry")
    print("checksum ok")
    if not paths.IMAGES.exists():
        with zipfile.ZipFile(paths.RAW_ZIP) as z:
            z.extractall(paths.RAW)
    print(f"{len(list(paths.IMAGES.glob('*.png')))} images in {paths.IMAGES}")


if __name__ == "__main__":
    main()
