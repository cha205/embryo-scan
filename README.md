# Embryo Scan

AI that watches 5-day IVF embryo time-lapse videos, detects every cell division, ranks embryos by predicted viability, and grades the resulting blastocyst like an expert embryologist.

> Status: in development. Time-lapse division model training; grading and ranking results below are final for their stage. The demo app runs locally (see Run it).

## What it does
1. **Detects every cell division in a time-lapse video.** A model labels the development stage of every frame (2 cells, 3 cells, ... blastocyst), then a dynamic program finds the most likely sequence that never goes backwards, which gives each division's time in hours.
2. **Ranks embryos by viability from their first three days.** Division timings up to day 3 predict which embryos will form a blastocyst by day 5, the standard viability proxy in time-lapse research.
3. **Grades the blastocyst.** A second model grades day-5 photos on the Gardner scale (expansion, inner cell mass, trophectoderm), benchmarked against a panel of embryologists.
4. **Shows its work.** A Three.js app replays a real video next to a 3D "digital twin" that divides at the detected times, and heat maps show what the grading model looked at.

## Results

**Viability ranking from day 0-3 timings** (blastocyst by day 5; 97 videos from held-out patients; AUC with 95% CI, 0.5 = coin flip):

| Timings from | Ranking method | AUC |
|---|---|---|
| Embryologist's annotations | Published Meseguer 2011 categories (no training) | 0.68 (0.55-0.78) |
| Embryologist's annotations | Logistic regression on timings | 0.72 (0.61-0.81) |
| AI-detected (end to end) | both | *training* |

The strongest signal is the length of the second cell cycle (cc2) and direct cleavage (1 -> 3 cells in under 5 h), both known warning signs in the literature.

**Division detection** (held-out patients): *training*.

**Grading vs experts** (300-photo gold test set graded by a panel of embryologists; accuracy):

| | Expansion | Inner cell mass | Trophectoderm |
|---|---|---|---|
| Individual experts (paper) | 0.78 | 0.74 | 0.70 |
| Best published model (paper) | 0.85 | 0.69 | 0.62 |
| **Embryo Scan, 5-model ensemble** | **0.853** (0.81-0.89) | **0.714** (0.66-0.77) | **0.660** (0.60-0.72) |
| Embryo Scan, balanced thresholds | 0.853 | 0.697 | 0.672 |

Our evaluation code reproduces the paper's published XCeption numbers exactly (0.78 / 0.69 / 0.62) before scoring our own model. The model's tendency to call grade-B inner cell masses "A" is inherited from the single embryologist who labeled its training data (they do the same on the test set); its weakness on rare grade-C trophectoderm is its own, and balanced decision thresholds tuned on training data raise grade-C recall from 3/30 to 12/30. Details: [reports/grading.md](reports/grading.md).

**Live birth from a photo + clinical data** (603 transferred embryos with known outcomes, patient-level CV): age alone AUC 0.53, embryologist's grade 0.55, AI grade 0.56, fine-tuned network 0.54. In this clinic's data live birth is barely predictable from any of them, and the project reports that rather than overclaiming. Details: [reports/outcome_lb.md](reports/outcome_lb.md).

## Data
- **Time-lapse videos:** Gomez et al. 2022, *A time-lapse embryo dataset for morphokinetic parameter prediction*, Data in Brief. 704 videos (311k frames, one every ~15 min for up to 5-6 days) with 16 phase annotations per video, frame timestamps and blastocyst grades for 160 videos. [Zenodo 10.5281/zenodo.7912264](https://doi.org/10.5281/zenodo.7912264), CC BY-NC-SA 4.0.
- **Blastocyst photos with outcomes:** Kromp et al. 2023, Scientific Data. 2,344 day-5 photos, 300 graded by an expert panel, 754 transferred embryos with live-birth outcomes. [figshare 10.6084/m9.figshare.20123153.v3](https://doi.org/10.6084/m9.figshare.20123153.v3), CC BY 4.0.

Data problems found and fixed along the way: spreadsheet-mangled hormone values (`1.26` stored as `Jän.26`), inconsistent file names, a test set split by image rather than patient, truncated JPEGs in the video archive, and skipped division phases that broke a published scoring rule until handled correctly.

## How it works
- **Division model:** ResNet-34 on a 3-frame stack (45 minutes before, now, 45 minutes after) so it can see cells mid-division; GPU augmentation (any rotation, flips, zoom, lighting); patient-level train/val/test split; monotonic dynamic-programming decoding.
- **Viability ranking:** division times up to 72 h -> logistic regression, compared with the published Meseguer 2011 hierarchy.
- **Grading model:** ConvNeXt-Tiny with three heads, 5-fold cross-fitting (every photo graded by a model that never saw it), test-time augmentation over 8 rotations/flips, Grad-CAM heat maps.
- **App:** Vite + Three.js front end; FastAPI grading endpoint (`app/server.py`).



## Run it
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt && pip install -e .
python scripts/download_data.py          # blastocyst photos, 624 MB
python scripts/download_timelapse.py --delete-archives   # time-lapse videos, 12 GB
python scripts/prepare_data.py && python scripts/prepare_timelapse.py
python scripts/train_grader.py --run convnext_t           # ~35 min on an RTX 2060
python scripts/train_timelapse.py --run r34
python scripts/viability.py --run r34
cd web && npm install && npm run dev      # app at http://localhost:5173, prototypes at /prototypes/
```

## Limitations
- The video dataset has no pregnancy outcomes, so viability means blastocyst formation by day 5, not pregnancy.
- Each dataset comes from a single clinic; results may not transfer to other clinics, microscopes or labeling styles.
- The 3D embryo is an illustration driven by detected timings, not a reconstruction of the cells.
- Research project, not a medical device.

## Plan and log

