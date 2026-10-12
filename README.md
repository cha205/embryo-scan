# Embryo Scan

Embryo Scan watches 5-day time-lapse videos of IVF embryos, detects every cell division, and ranks the embryos by how likely they are to develop. It also grades day-5 embryo photos on the scale embryologists use. The web app replays real videos next to a 3D model of the embryo and can grade a photo you upload.

[![CI](https://github.com/cha205/embryo-scan/actions/workflows/ci.yml/badge.svg)](https://github.com/cha205/embryo-scan/actions/workflows/ci.yml)

Try it at [embryo-scan.vercel.app](https://embryo-scan.vercel.app).

![The development stage of one held-out embryo over five days, from the embryologist's labels and from the model](reports/figures/timelapse-stage-timeline.png)

Every number below comes from patients the models never saw in training.

## Results

### Detecting cell divisions

On 141 test videos from 108 patients (59k labelled frames), the model gets the development stage of 71.8% of frames exactly right and 92.8% within one stage. Its division times are off by a median of 1 hour from the embryologist's marks, with frames 15 minutes apart.

| Division | 2 cells | 3 cells | 4 cells | 5 cells | 8 cells | morula | blastocyst |
|---|---|---|---|---|---|---|---|
| Median error (h) | 0.2 | 0.3 | 0.5 | 1.0 | 1.7 | 1.7 | 1.1 |
| Within 2 h | 94% | 84% | 87% | 70% | 56% | 56% | 68% |

Later stages are harder because the cells overlap, which makes them hard to count for embryologists too. Full results are in [divisions-test.md](reports/timelapse/divisions-test.md), with the error spread in [this plot](reports/figures/timelapse-timing-errors.png).

### Deep learning vs classical image processing

To see what the neural network adds, I also built a version with no machine learning in MATLAB. It finds the embryo with a circular Hough transform, measures how much it changes between frames, and turns motion spikes into division times using typical embryo timing. I checked the rules on the validation videos and ran the same 141 test videos once.

| Videos timed within 1 h | 2 cells | 3 cells | 4 cells | 5 cells |
|---|---|---|---|---|
| Classical (MATLAB) | 44% | 34% | 45% | 22% |
| Deep model (ResNet-34) | 92% | 78% | 76% | 57% |

A motion detector also fires when cells shift or the embryo drifts, while the network learned what a division looks like. Details: [classical-baseline-test.md](reports/timelapse/classical-baseline-test.md) and [matlab/](matlab).

### Ranking embryos by viability

From division times up to day 3, the model predicts which embryos reach the blastocyst stage by day 5, the usual viability measure in time-lapse research. On the 97 test videos recorded that long (AUC with a 95% confidence interval, where 0.5 is a coin flip):

| Timings from | Ranking method | AUC |
|---|---|---|
| Embryologist's annotations | Published Meseguer 2011 categories | 0.68 (0.55-0.78) |
| Embryologist's annotations | Logistic regression | 0.72 (0.61-0.81) |
| The model, from raw video | Logistic regression | 0.69 (0.57-0.79) |
| The model, from raw video | Published Meseguer 2011 categories | 0.61 (0.48-0.73) |

Straight from the video, the ranking is almost as good as one built on an embryologist's hand annotations. The strongest signals are the length of the second cell cycle and direct cleavage (1 to 3 cells in under 5 hours), both known warning signs. Details: [viability.md](reports/timelapse/viability.md) and the [ROC curves](reports/figures/timelapse-viability-roc.png).

### Grading blastocyst photos

Accuracy on 300 photos graded by a panel of embryologists:

| | Expansion | Inner cell mass | Trophectoderm |
|---|---|---|---|
| Individual embryologists (paper) | 0.78 | 0.74 | 0.70 |
| Best published model (paper) | 0.85 | 0.69 | 0.62 |
| Embryo Scan, 5-model ensemble | 0.853 (0.81-0.89) | 0.714 (0.66-0.77) | 0.660 (0.60-0.72) |
| Embryo Scan, balanced thresholds | 0.853 | 0.697 | 0.672 |

Before scoring my model, I checked that my evaluation code reproduces the paper's published numbers exactly (0.78 / 0.69 / 0.62). The test set was picked by image rather than by patient, so 30% of the training photos come from test patients. Retraining without them changes accuracy by less than 0.01 against a size-matched control, so the overlap isn't inflating these numbers ([patient-overlap-check.md](reports/blastocyst/patient-overlap-check.md)).

The model calls too many grade-B inner cell masses "A", but so does the one embryologist who labelled its training data. Rare grade-C trophectoderm is the model's own weak spot; balanced thresholds tuned on the training data raise its recall from 3/30 to 12/30. Details: [grading.md](reports/blastocyst/grading.md).

### Predicting live birth

For the 603 transferred embryos with known outcomes, live birth was barely predictable from anything. Age alone gets an AUC of 0.53, the embryologist's grade 0.55, the model's grade 0.56 and a fine-tuned network 0.54. Details: [outcome-live-birth.md](reports/blastocyst/outcome-live-birth.md).

## How it works

- The division model is a ResNet-34 that sees three frames at once (45 minutes before, now and 45 minutes after), so it can catch cells mid-division. A dynamic program then finds the most likely sequence of stages that never goes backwards, which turns noisy per-frame guesses into clean division times.
- The viability ranking is a logistic regression on division times up to 72 hours, compared against the published Meseguer 2011 categories.
- The grading model is a ConvNeXt-Tiny with three heads, trained with 5-fold cross-fitting so every photo is graded by a model that never saw it. It averages 8 rotations and flips at test time, and Grad-CAM shows where it looked.
- My own splits are all by patient, because sibling embryos look alike.
- The web app uses Three.js and Vite, and a FastAPI server grades uploaded photos.

## Data

- Time-lapse videos: Gomez et al. 2022, *A time-lapse embryo dataset for morphokinetic parameter prediction*, Data in Brief. 704 videos (311k frames, one every 15 minutes for up to 6 days) with 16 phase annotations each. [Zenodo](https://doi.org/10.5281/zenodo.7912264), CC BY-NC-SA 4.0.
- Blastocyst photos: Kromp et al. 2023, Scientific Data. 2,344 day-5 photos, 300 graded by an expert panel and 754 transferred embryos with live-birth outcomes. [figshare](https://doi.org/10.6084/m9.figshare.20123153.v3), CC BY 4.0.

The raw data needed fixing first. A spreadsheet had turned hormone values into dates (`1.26` became `Jän.26`), file names were padded inconsistently, some JPEGs were cut off, timestamps were rounded so neighbouring frames shared one, and skipped division phases broke a published scoring rule until I handled them. [timelapse-data-tour.ipynb](notebooks/timelapse-data-tour.ipynb) walks through the video data.

## Tech stack

| Area | Tools |
|---|---|
| Deep learning | PyTorch (CUDA, mixed precision), timm, Grad-CAM |
| Classical image processing | MATLAB with the Image Processing Toolbox |
| Data and statistics | pandas, NumPy, SciPy, scikit-learn, LightGBM |
| Images, video and figures | OpenCV, Pillow, matplotlib, Jupyter |
| App | Three.js, Vite, FastAPI |
| Tooling | pytest, MATLAB unit tests, ruff, GitHub Actions, Docker, uv |

## Repository layout

```
embryoscan/   Python package: blastocyst/, timelapse/ and common/ (paths.py lists every file location)
scripts/      one command per step, for each pipeline
matlab/       classical image-processing baseline
server/       FastAPI photo grader
web/          Three.js app and design prototypes
tests/        pytest and MATLAB unit tests
notebooks/    data exploration
reports/      results and figures written by the scripts
splits/       fixed patient-level train/val/test splits
deploy/       hosting the demo (Hugging Face Space, model card)
```

Downloads go to `data/` and trained models to `runs/`; both are rebuilt by the scripts and not committed.

## Running it

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt && pip install -e .

# Time-lapse videos (12 GB) -> division model -> viability
python scripts/timelapse/download_data.py --delete-archives
python scripts/timelapse/prepare_data.py
python scripts/timelapse/train_stages.py                      # ~80 min on an RTX 2060
python scripts/timelapse/train_stages.py --eval-only --split test
python scripts/timelapse/run_viability.py
python scripts/timelapse/plot_divisions.py

# Blastocyst photos (624 MB) -> grading model
python scripts/blastocyst/download_data.py
python scripts/blastocyst/prepare_data.py
python scripts/blastocyst/train_grader.py                     # ~35 min
python scripts/blastocyst/report_grading.py

# Classical baseline and tests
matlab -batch "addpath('matlab'); runClassicalBaseline"
pytest
matlab -batch "assertSuccess(runtests('tests/matlab'))"

# Web app at http://localhost:5173 (the grading server is only needed to upload a photo)
uvicorn server.main:app --port 8000
cd web && npm install && npm run dev
```

Every script has `--help`. The live-birth analyses are in `scripts/blastocyst/run_baselines.py`, `run_outcome_models.py` and `finetune_outcome.py`. [deploy/README.md](deploy/README.md) covers putting the demo online.

## Limitations

- The video dataset has no pregnancy outcomes, so viability here means reaching the blastocyst stage by day 5, not pregnancy.
- Each dataset comes from a single clinic, so the models may not transfer to other clinics, microscopes or labelling styles.
- The 3D embryo is an illustration driven by the detected timings, not a reconstruction of the real cells.
- This is a research project, not a medical device.

## License

The code is MIT licensed. The data keeps its own licenses: the video frames in `web/public/timelapse/` are CC BY-NC-SA 4.0 (Gomez et al. 2022) and the photos in `web/public/blastocyst/` are CC BY 4.0 (Kromp et al. 2023).
