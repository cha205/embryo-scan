# Classical baseline (MATLAB)

A version of the division detector with no machine learning, built with MATLAB's Image Processing Toolbox. It runs on the same held-out test videos as the deep model and is scored with the same rules, so the gap between the two shows what the neural network adds.

| Step | Function | MATLAB tools |
|---|---|---|
| Load a video as an image stack | `classical.loadVideo` | `imread`, `rgb2gray`, `imresize` |
| Find the embryo | `classical.findEmbryo`, `classical.textureCenter` | `imfindcircles`, `stdfilt`, `imgaussfilt` |
| Measure frame-to-frame change | `classical.motionEnergy` | `imgaussfilt` |
| Turn motion spikes into t2-t5 | `classical.detectDivisions` | `movmean`, `islocalmax` |
| Score against the annotations | `classical.onsetsFromPhases`, `classical.meseguerCategory` | same rules as `embryoscan/timelapse/labels.py` |
| Report and figures | `reportClassicalBaseline` | `tiledlayout`, `exportgraphics` |

The division rules come from typical embryo timing. The first division is the strongest motion spike between 18 and 38 hours after fertilization, the next two follow 5 to 18 hours later (together if the cells divide in sync), and the fifth cell appears 6 to 20 hours after that. I checked the rules on the validation videos and ran the test set once.

## Running it

It needs MATLAB R2022a or newer with the Image Processing Toolbox (I used R2024a), plus the prepared time-lapse data and the deep model's test predictions:

```bash
python scripts/timelapse/prepare_data.py
python scripts/timelapse/train_stages.py --eval-only --split test
```

Then, from the repository root:

```bash
matlab -batch "addpath('matlab'); runClassicalBaseline"
```

The 141 test videos take about 10 minutes, mostly spent reading 60k frames. Per-video results go to `runs/timelapse/classical-matlab/`, the comparison to [classical-baseline-test.md](../reports/timelapse/classical-baseline-test.md) and the figures to `reports/figures/`. `runClassicalBaseline(Limit=3)` is a quick check on three videos, and `runClassicalBaseline(Split="val")` runs the validation videos.

## Tests

```bash
matlab -batch "assertSuccess(runtests('tests/matlab'))"
```

The tests in [tests/matlab/](../tests/matlab) use synthetic images, signals and a small fake video, so they don't need the dataset.
