"""Grades uploaded embryo photos and serves the built web app.

Run from the project root:
  uvicorn server.main:app --port 8000

On a GPU each photo is graded over all 8 rotations/flips, as in the evaluation.
On a CPU it uses 2 views for the grade and 1 per heat map, about 5 s instead of
20 s on two cores.

Environment variables:
  ALLOWED_ORIGINS    comma-separated origins allowed to call the API from
                     another domain, e.g. https://embryo-scan.vercel.app
  EMBRYOSCAN_DEVICE  "cpu" to keep the GPU free
"""

from __future__ import annotations

import base64
import io
import os
import threading
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from embryoscan import paths
from embryoscan.blastocyst.inference import grade, heatmap, load_ensemble, load_image

MAX_BYTES = 10 * 1024 * 1024
DEVICE = os.environ.get("EMBRYOSCAN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
GRADE_VIEWS, HEATMAP_VIEWS = (8, 8) if DEVICE == "cuda" else (2, 1)
state: dict = {}
grading = threading.Lock()  # Grad-CAM registers hooks on the shared models


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["models"] = load_ensemble(device=DEVICE)
    yield
    state.clear()


app = FastAPI(title="Embryo Scan", lifespan=lifespan)
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


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
        res = grade(models, x, views=GRADE_VIEWS)
        cams = {}
        for head, code in res["codes"].items():
            cls = code if head == "exp" or code < 3 else 0  # ungraded ICM/TE: show class A
            cams[head] = heatmap(models[0], x, head, cls, views=HEATMAP_VIEWS)
    res["photo"] = png_data_url(Image.fromarray(x[0, 0].numpy()))
    res["heatmaps"] = {
        head: png_data_url(Image.fromarray((cam * 255).astype(np.uint8)).resize((192, 144), Image.Resampling.BILINEAR))
        for head, cam in cams.items()
    }
    res["views"] = GRADE_VIEWS
    return res


@app.get("/api/health")
def health():
    return {"ok": True, "models": len(state.get("models", [])), "views": GRADE_VIEWS}


dist = paths.ROOT / "web" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")
