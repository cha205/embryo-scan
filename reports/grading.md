# Grading results

Gold test set: 300 images graded by a panel of embryologists (majority vote). No grading model trained on these images. Accuracy is computed exactly like the dataset authors' script: inner cell mass and trophectoderm are only scored when neither the model nor the panel puts the embryo at expansion grade 1-2.

|  | Expansion | Inner cell mass | Trophectoderm |
|---|---|---|---|
| Individual experts vs panel (paper, mean ± SD) | 0.78 ± 0.12 | 0.74 ± 0.06 | 0.70 ± 0.14 |
| Best published model (paper) | 0.85 Swin | 0.69 XCeption | 0.62 XCeption/Swin |
| Paper's XCeption, scored by our code | 0.780 | 0.686 | 0.616 |
| Embryo Scan, single models (mean, range) | 0.831 (0.813-0.853) | 0.706 (0.691-0.718) | 0.654 (0.647-0.667) |
| **Embryo Scan, 5-model ensemble** (95% CI) | **0.853** (0.81-0.89) | **0.714** (0.66-0.77) | **0.660** (0.60-0.72) |

Ensemble weighted F1: expansion 0.852, ICM 0.672, TE 0.608. Expansion quadratic-weighted kappa: 0.916.

Caveat: the official test set was drawn by image, not by patient, so some training images come from the same patients as test images (same as the paper's setup). A patient-disjoint check is planned.

## Rare grades: balanced decision thresholds

Grade C is rare in training (16 ICM and 47 TE labels out of ~1,700), so the plain model almost never predicts it. Adding a per-class bonus to the B and C scores, tuned to maximize balanced accuracy on the **out-of-fold training predictions** (the gold set is not used), trades a little overall accuracy for catching rare grades. Bonuses (log-prob, A/B/C): ICM [0.0, 0.0, 0.9], TE [0.0, 0.1, 1.3].

| decision rule | ICM: accuracy / macro F1 / C found | TE: accuracy / macro F1 / C found |
|---|---|---|
| argmax (as trained) | 0.714 / 0.488 / 1 of 16 | 0.660 / 0.525 / 3 of 30 |
| balanced thresholds | 0.697 / 0.489 / 2 of 16 | 0.672 / 0.615 / 12 of 30 |

## Where the model and the panel disagree

Expansion grade:

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