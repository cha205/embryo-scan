"""Embryo Scan demo server: grades uploaded embryo photos and serves the web app.

Run from the project root:
  uvicorn server.main:app --port 8000
"""

from __future__ import annotations

import base64
import io
import threading
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from embryoscan import paths
from embryoscan.blastocyst.inference import grade, heatmap, load_ensemble, load_image

MAX_BYTES = 10 * 1024 * 1024
state: dict = {}
grading = threading.Lock()  # Grad-CAM hooks into the shared models, so one request at a time


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["models"] = load_ensemble()
    yield
    state.clear()


app = FastAPI(title="Embryo Scan", lifespan=lifespan)


def png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/grade")
def api_grade(file: Annotated[UploadFile, File()]):
    """Grade one photo. Returns the grade, class probabilities per head, the
    photo exactly as the model saw it (grayscale, cropped to 4:3, 384 x 288)
    and a Grad-CAM heat map per head, aligned with that photo."""
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "That image is larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file isn't an image we can read. Try a PNG or JPG.") from None
    models = state["models"]
    x = load_image(img)
    with grading:
        res = grade(models, x)
        cams = {}
        for head, code in res["codes"].items():
            cls = code if head == "exp" or code < 3 else 0  # ICM/TE aren't graded early: show class A
            cams[head] = heatmap(models[0], x, head, cls)
    res["photo"] = png_data_url(Image.fromarray(x[0, 0].numpy()))
    res["heatmaps"] = {
        head: png_data_url(Image.fromarray((cam * 255).astype(np.uint8)).resize((192, 144), Image.Resampling.BILINEAR))
        for head, cam in cams.items()
    }
    return res


@app.get("/api/health")
def health():
    return {"ok": True, "models": len(state.get("models", []))}


dist = paths.ROOT / "web" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")
