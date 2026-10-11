"""Embryo Scan demo server: grades uploaded embryo photos and serves the web app.

Run from the project root:
  uvicorn app.server:app --port 8000
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from embryoscan import paths
from embryoscan.inference import grade, heatmap, load_ensemble, load_image

MAX_BYTES = 10 * 1024 * 1024
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["models"] = load_ensemble()
    yield
    state.clear()


app = FastAPI(title="Embryo Scan", lifespan=lifespan)


def png_b64(cam: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((cam * 255).astype(np.uint8)).resize((192, 144), Image.Resampling.BILINEAR).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/grade")
async def api_grade(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file isn't an image we can read. Try a PNG or JPG.")
    models = state["models"]
    x = load_image(img)
    res = grade(models, x)
    res["heatmaps"] = {}
    for head, code in res["codes"].items():
        cls = code if head == "exp" or code < 3 else 0
        res["heatmaps"][head] = png_b64(heatmap(models[0], x, head, cls))
    return res


@app.get("/api/health")
def health():
    return {"ok": True, "models": len(state.get("models", []))}


dist = paths.ROOT / "web" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")
