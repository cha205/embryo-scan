# Embryo Scan

AI that grades IVF embryos from microscope photos like an expert embryologist, and tests honestly whether the photo helps predict live birth.

> Status: in development. Grading model done; outcome experiments and demo app in progress. Outcome numbers are cross-validation on development data; the locked test set hasn't been opened yet.

## What it does
- **Grades the embryo.** A day-5 blastocyst photo goes in; out comes a Gardner grade (expansion 1-5, inner cell mass A-C, trophectoderm A-C), the same scale embryologists use.
- **Predicts live birth.** Combines the photo with clinical data (age, AMH, endometrium, egg counts) and compares against what age alone, or the embryologist's grade alone, can do.
- **Shows its reasoning.** Grad-CAM heat maps show which part of the embryo drove each grade.

## Results so far

**Grading vs experts** (300-photo gold test set, graded by a panel of embryologists; accuracy):

| | Expansion | Inner cell mass | Trophectoderm |
|---|---|---|---|
| Individual experts (paper) | 0.78 | 0.74 | 0.70 |
| Best published model (paper) | 0.85 | 0.69 | 0.62 |
| **Embryo Scan, 5-model ensemble** | **0.853** (0.81-0.89) | **0.714** (0.66-0.77) | **0.660** (0.60-0.72) |

Our evaluation code reproduces the paper's published XCeption numbers exactly (0.78 / 0.69 / 0.62) before scoring our own model. Full breakdown with confusion matrices: [reports/grading.md](reports/grading.md). The main error is optimism: the model calls many grade-B inner cell masses "A", because 77% of its training labels are A.

**Live birth** (603 embryos, patient-level 5-fold CV, AUC with 95% CI; 0.5 = coin flip):

| Model | AUC |
|---|---|
| Age only | 0.53 (0.48-0.58) |
| Embryologist's grade | 0.55 (0.50-0.60) |
| All clinical data | 0.47 (0.42-0.52) |
| AI grade (from the photo) | 0.56 (0.51-0.61) |
| AI grade + embryologist's grade + age | 0.56 (0.51-0.61) |

Honest finding so far: in this clinic's data, live birth is barely predictable from age, the embryologist's grade, or the photo. The AI's grade does as well as the embryologist's, but no model clearly beats age alone (paired differences all include zero). Details: [reports/outcome_lb.md](reports/outcome_lb.md).

## Data
Kromp et al. 2023, *An annotated human blastocyst dataset to benchmark deep learning architectures for in vitro fertilization*, Scientific Data. 2,344 day-5 photos (851 patient IDs in the files; the paper reports 837 patients), 754 transferred embryos with outcomes. [figshare](https://doi.org/10.6084/m9.figshare.20123153.v3), CC BY 4.0.

Cleaning that mattered (details in [reports/data_report.md](reports/data_report.md)):
- 287 of 589 AMH hormone values had been turned into dates by a spreadsheet (`Jän.26` = 1.26, `01.Apr` = 1.4). Decoded from the conversion rules; 118 ambiguous values tested both ways.
- Image names padded inconsistently across files; 22 double-embryo transfers share one outcome label.
- The official grading test set was drawn by image, not by patient (511 training images share a patient with a test image).

## How it works
1. **Grading model:** ConvNeXt-Tiny (ImageNet-pretrained) with three heads, trained with 5-fold cross-fitting so every photo gets a grade from a model that never saw it. Augmentation (any rotation, flips, zoom, lighting) runs on the GPU; predictions average 8 rotations/flips.
2. **Outcome model:** logistic regression on the grading model's outputs plus clinical data, evaluated with patient-level cross-validation and paired bootstrap comparisons.
3. **Demo:** a web app with a Three.js front end. Three 3D design prototypes are in [web/prototypes](web/prototypes).



## Run it
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt && pip install -e .
python scripts/download_data.py      # 624 MB
python scripts/prepare_data.py       # cleaning, splits, image cache, data report
python scripts/check_paper_eval.py   # reproduces the paper's numbers
python scripts/baselines.py          # clinical baselines
python scripts/train_grader.py --run convnext_t   # ~35 min on an RTX 2060
python scripts/outcome_models.py
cd web && npm install && npm run dev  # 3D prototypes at /prototypes/
```

## Limitations
- One clinic, one microscope, one embryologist's training labels: results may not transfer to other clinics.
- 754 outcomes is small; confidence intervals are wide.
- The 3D embryo in the demo is an illustration generated from the predicted grade, not a reconstruction of the photo.
- Research project, not a medical device.

## Plan and log

