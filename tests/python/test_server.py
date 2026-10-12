import base64
import io

import pytest
from PIL import Image

from embryoscan import paths
from embryoscan.blastocyst.grading import DEFAULT_RUN

pytestmark = pytest.mark.skipif(
    not (paths.run_dir("blastocyst", DEFAULT_RUN) / "fold-4.pt").exists() or not paths.KROMP_IMAGES.exists(),
    reason="needs trained grading models and the dataset",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from server.main import app

    with TestClient(app) as c:  # loads the 5-model ensemble once
        yield c


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_grade_endpoint_returns_grade_photo_and_heatmaps(client):
    r = client.post("/api/grade", files={"file": ("embryo.png", png(Image.open(paths.KROMP_IMAGES / "0057_01.png")))})
    assert r.status_code == 200
    body = r.json()
    assert body["grade"] == "4AA"  # the expert panel's grade for this gold-test image
    assert set(body["heatmaps"]) == {"exp", "icm", "te"}
    assert abs(sum(body["probs"]["exp"]) - 1) < 1e-4
    photo = Image.open(io.BytesIO(base64.b64decode(body["photo"].split(",", 1)[1])))
    assert photo.size == (384, 288)  # exactly what the model saw, so the heat maps line up


def test_grade_endpoint_takes_other_shapes(client):
    tall = Image.new("L", (512, 683), 128)  # portrait, like a phone photo
    tall.paste(Image.open(paths.KROMP_IMAGES / "0057_01.png").convert("L"), (0, 150))
    r = client.post("/api/grade", files={"file": ("tall.png", png(tall))})
    assert r.status_code == 200 and r.json()["grade"]


def test_grade_endpoint_rejects_bad_uploads(client):
    bad = client.post("/api/grade", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert bad.status_code == 400
    huge = client.post("/api/grade", files={"file": ("huge.png", b"0" * (10 * 1024 * 1024 + 1), "image/png")})
    assert huge.status_code == 413
