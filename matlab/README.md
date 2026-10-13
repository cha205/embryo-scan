# Classical baseline (MATLAB)

This directory contains a version of the division detector that uses no machine learning and is built with the MATLAB Image Processing Toolbox. It is applied to the same held-out test videos as the deep model and scored with the same rules, so the difference between the two quantifies the contribution of the neural network.

| Step | Function | MATLAB tools |
|---|---|---|
| Load a video as an image stack | `classical.loadVideo` | `imread`, `rgb2gray`, `imresize` |
| Locate the embryo | `classical.findEmbryo`, `classical.textureCenter` | `imfindcircles`, `stdfilt`, `imgaussfilt` |
| Measure frame-to-frame change | `classical.motionEnergy` | `imgaussfilt` |
| Convert motion spikes into t2-t5 | `classical.detectDivisions` | `movmean`, `islocalmax` |
| Score against the annotations | `classical.onsetsFromPhases`, `classical.meseguerCategory` | same rules as `embryoscan/timelapse/labels.py` |
| Report and figures | `reportClassicalBaseline` | `tiledlayout`, `exportgraphics` |

The division rules are derived from typical embryo timing. The first division is assigned to the strongest motion spike between 18 and 38 hours after insemination, the next two divisions to spikes 5 to 18 hours later (a single spike when the cells divide synchronously), and the appearance of the fifth cell to a spike 6 to 20 hours after that. The rules were checked on the validation videos, and the test videos were processed once.

## Running the baseline

The baseline requires MATLAB R2022a or newer with the Image Processing Toolbox (the reported results were produced with R2024a), together with the prepared time-lapse data and the deep model's test predictions:

```bash
python scripts/timelapse/prepare_data.py
python scripts/timelapse/train_stages.py --eval-only --split test
```

The baseline is then run from the repository root:

```bash
matlab -batch "addpath('matlab'); runClassicalBaseline"
```

Processing the 141 test videos takes about 10 minutes, most of which is spent reading 60,000 frames. Per-video results are written to `runs/timelapse/classical-matlab/`, the comparison to [classical-baseline-test.md](../reports/timelapse/classical-baseline-test.md) and the figures to `reports/figures/`. `runClassicalBaseline(Limit=3)` performs a quick check on three videos, and `runClassicalBaseline(Split="val")` processes the validation videos.

## Tests

```bash
matlab -batch "assertSuccess(runtests('tests/matlab'))"
```

The tests in [tests/matlab/](../tests/matlab) use synthetic images, signals and a small synthetic video, so they do not require the dataset.
