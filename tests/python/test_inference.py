import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")  # Grad-CAM needs OpenCV, which needs system graphics libraries
from embryoscan.blastocyst.inference import load_image  # noqa: E402


def test_dataset_sized_photos_are_only_resized():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (384, 512), dtype=np.uint8))
    expected = np.asarray(img.resize((384, 288), Image.Resampling.LANCZOS))
    assert np.array_equal(load_image(img)[0, 0].numpy(), expected)


def test_other_shapes_are_cropped_not_stretched():
    # a round embryo in the middle of a tall phone photo
    yy, xx = np.mgrid[:800, :600]
    disk = (np.hypot(xx - 300, yy - 400) < 150).astype(np.uint8) * 255
    out = load_image(Image.fromarray(disk))[0, 0].numpy() > 127
    rows, cols = np.flatnonzero(out.any(1)), np.flatnonzero(out.any(0))
    assert out.shape == (288, 384)
    assert abs(np.ptp(rows) - np.ptp(cols)) <= 2  # still round
    assert abs(rows.mean() - 143.5) <= 2 and abs(cols.mean() - 191.5) <= 2  # still centred
