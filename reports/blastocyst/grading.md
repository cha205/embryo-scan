# Grading results

The gold test set comprises 300 images graded by a panel of embryologists (majority vote), none of which was used to train a grading model. Accuracy is computed exactly as in the dataset authors' scoring script, in which the inner cell mass and the trophectoderm are scored only when neither the model nor the panel assigns expansion grade 1-2.

| Grader | Expansion | Inner cell mass | Trophectoderm |
|---|---|---|---|
| Individual embryologists vs. panel (Kromp et al., 2023; mean ± SD) | 0.78 ± 0.12 | 0.74 ± 0.06 | 0.70 ± 0.14 |
| Best published model (Kromp et al., 2023) | 0.85 (Swin) | 0.69 (Xception) | 0.62 (Xception/Swin) |
| Published Xception predictions, re-scored with this code | 0.780 | 0.686 | 0.616 |
| Embryo Scan, single models (mean, range) | 0.831 (0.813-0.853) | 0.706 (0.691-0.718) | 0.654 (0.647-0.667) |
| **Embryo Scan, five-model ensemble** (95% CI) | **0.853** (0.81-0.89) | **0.714** (0.66-0.77) | **0.660** (0.60-0.72) |

The weighted F1 scores of the ensemble are 0.852 for expansion, 0.672 for the inner cell mass and 0.608 for the trophectoderm, and the quadratically weighted kappa for expansion is 0.916.

The official test set was drawn by image rather than by patient, so 30% of the training images come from patients who also contribute a test image, as in the original benchmark. Retraining without those patients changes accuracy by less than 0.01 relative to a size-matched control ([patient-overlap-check.md](patient-overlap-check.md)).

## Balanced decision thresholds for rare grades

Grade C is rare in the training data (16 inner cell mass and 47 trophectoderm labels among approximately 1,700 images), and the argmax decision rule therefore seldom predicts it. Adding a class-specific offset to the log-probabilities of grades B and C, tuned to maximize balanced accuracy on the **out-of-fold training predictions** without use of the gold set, exchanges a small loss of overall accuracy for improved detection of rare grades. The offsets (log-probability, grades A/B/C) are ICM [0.0, 0.0, 0.9] and TE [0.0, 0.1, 1.3].

| decision rule | ICM: accuracy / macro F1 / grade C detected | TE: accuracy / macro F1 / grade C detected |
|---|---|---|
| argmax | 0.714 / 0.488 / 1 of 16 | 0.660 / 0.525 / 3 of 30 |
| balanced thresholds | 0.697 / 0.489 / 2 of 16 | 0.672 / 0.615 / 12 of 30 |

## Confusion matrices against the panel

Expansion grade (rows: panel; columns: model):

| panel \ model | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | 20 | 3 | 0 | 0 | 0 |
| 2 | 3 | 17 | 11 | 0 | 0 |
| 3 | 0 | 5 | 74 | 7 | 0 |
| 4 | 0 | 0 | 8 | 141 | 4 |
| 5 | 0 | 0 | 1 | 0 | 4 |
| n/a | 0 | 0 | 0 | 0 | 2 |

Inner cell mass:

| panel \ model | A | B | C | n/d |
|---|---|---|---|---|
| A | 142 | 8 | 1 | 2 |
| B | 35 | 29 | 2 | 2 |
| C | 4 | 11 | 1 | 1 |
| n/d | 5 | 5 | 1 | 43 |
| n/a | 2 | 6 | 0 | 0 |

Trophectoderm:

| panel \ model | A | B | C | n/d |
|---|---|---|---|---|
| A | 114 | 8 | 0 | 0 |
| B | 35 | 42 | 0 | 5 |
| C | 0 | 27 | 3 | 0 |
| n/d | 4 | 7 | 0 | 43 |
| n/a | 4 | 8 | 0 | 0 |