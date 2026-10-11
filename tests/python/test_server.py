import io

import pytest
from PIL import Image

from embryoscan import paths
from embryoscan.blastocyst.grading import DEFAULT_RUN

pytestmark = pytest.mark.skipif(
    not (paths.run_dir("blastocyst", DEFAULT_RUN) / "fold-4.pt").exists() or not paths.KROMP_IMAGES.exists(),
    reason="needs trained grading models and the dataset",
)


def test_grade_endpoint_returns_grade_and_heatmaps():
    from fastapi.testclient import TestClient

    from server.main import app

    buf = io.BytesIO()
    Image.open(paths.KROMP_IMAGES / "0057_01.png").save(buf, "PNG")
    with TestClient(app) as client:
        r = client.post("/api/grade", files={"file": ("embryo.png", buf.getvalue(), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert body["grade"] == "4AA"  # the expert panel's grade for this gold-test image
        assert set(body["heatmaps"]) == {"exp", "icm", "te"}
        assert abs(sum(body["probs"]["exp"]) - 1) < 1e-4

        bad = client.post("/api/grade", files={"file": ("notes.txt", b"not an image", "text/plain")})
        assert bad.status_code == 400
