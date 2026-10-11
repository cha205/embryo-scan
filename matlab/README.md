# MATLAB: classical image-processing baseline

How well can cell divisions be timed *without* machine learning? This folder answers that with a hand-built pipeline in MATLAB's Image Processing Toolbox, run on the same held-out test videos as the deep model and scored with the same rules. The gap between the two is what the neural network adds.

| step | function | MATLAB tools |
|---|---|---|
| Load a video as an image stack | `classical.loadVideo` | `imread`, `rgb2gray`, `imresize` |
| Find the embryo | `classical.findEmbryo`, `classical.textureCenter` | `imfindcircles` (circular Hough transform), `stdfilt`, `imgaussfilt` |
| Measure frame-to-frame change | `classical.motionEnergy` | `imgaussfilt`, robust z-score |
| Turn motion spikes into t2-t5 | `classical.detectDivisions` | `movmean`, `islocalmax` |
| Score against the annotations | `classical.onsetsFromPhases`, `classical.meseguerCategory` | mirror `embryoscan/timelapse/labels.py` |
| Report and figures | `reportClassicalBaseline` | `tiledlayout`, `exportgraphics` |

The rules in `detectDivisions` come from embryology, not from fitting: the first division is the strongest motion spike between 18 and 38 hours after fertilization, the next two follow 5 to 18 hours later (together if the cells split in sync), and the fifth cell appears 6 to 20 hours after that. They were checked on the validation videos only; the test videos were run once.

## Run

Needs MATLAB R2022a or newer (developed on R2024a) with the Image Processing Toolbox, plus the prepared time-lapse data and the deep model's test predictions:

```bash
python scripts/timelapse/prepare_data.py
python scripts/timelapse/train_stages.py --eval-only --split test
```

Then, from the repository root:

```bash
matlab -batch "addpath('matlab'); runClassicalBaseline"
```

That times all 141 test videos (about 10 minutes, mostly reading 60k frames) and writes:

- `runs/timelapse/classical-matlab/`: per-video division times, motion-energy traces and a log (not committed)
- [`reports/timelapse/classical-baseline-test.md`](../reports/timelapse/classical-baseline-test.md): the comparison
- `reports/figures/timelapse-classical-*.png`: the figures

`runClassicalBaseline(Limit=3)` is a quick check on three videos, and `runClassicalBaseline(Split="val")` runs the validation videos.

## Tests

```bash
matlab -batch "assertSuccess(runtests('tests/matlab'))"
```

Class-based `matlab.unittest` tests in [`tests/matlab/`](../tests/matlab) run on synthetic images, signals and a fake video written to a temporary folder, so they need no data.

## Layout

```
matlab/
  runClassicalBaseline.m      entry point: detect, save, report
  reportClassicalBaseline.m   scoring, markdown report, figures
  +classical/                 the pipeline, one function per file
tests/matlab/                 one test class per function
```
