"""Patient-level splits: all of a patient's images go to the same split, because
sibling embryos share genetics and imaging conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 2026
N_FOLDS = 5


def _patient_folds(labels: pd.Series, patients: pd.Series, n_splits: int, seed: int) -> np.ndarray:
    """Fold per row, keeping each patient in one fold and the label rate similar across folds."""
    folds = np.full(len(labels), -1)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(cv.split(np.zeros(len(labels)), labels.to_numpy(), patients.to_numpy())):
        folds[idx] = k
    return folds


def make_outcome_split(clinical: pd.DataFrame, target: str = "lb", seed: int = SEED) -> pd.DataFrame:
    """Hold out about 20% of patients as the test set and split the rest into 5
    cross-validation folds. Returns file, patient, split ('dev' or 'test') and
    fold (0-4, or -1 for test)."""
    first = _patient_folds(clinical[target], clinical["patient"], N_FOLDS, seed)
    out = clinical[["file", "patient"]].copy()
    out["split"] = np.where(first == 0, "test", "dev")
    dev = out["split"] == "dev"
    out["fold"] = -1
    out.loc[dev, "fold"] = _patient_folds(clinical.loc[dev, target], clinical.loc[dev, "patient"], N_FOLDS, seed + 1)
    return out


def make_grade_split(grades: pd.DataFrame, outcome_split: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Split the graded images for training the grading model.

    split is one of:
      pool       silver-labelled training images in 5 patient-level folds; one
                 model per fold, so every pool image is graded out-of-fold
      gold_test  the 300 expert-panel images, used only for evaluation
      held_out   images of patients in the outcome test set, which no model sees
      unused     the image with conflicting labels and the unlabelled one
    gold_patient marks patients who also have a gold image.
    """
    test_patients = set(outcome_split.loc[outcome_split.split == "test", "patient"])
    gold_patients = set(grades.loc[grades.label_set == "gold", "patient"])
    out = grades[["file", "patient", "label_set"]].copy()
    out["split"] = "unused"
    out.loc[out.label_set == "gold", "split"] = "gold_test"
    silver = out.label_set == "silver"
    out.loc[silver & out.patient.isin(test_patients), "split"] = "held_out"
    pool = silver & ~out.patient.isin(test_patients)
    out.loc[pool, "split"] = "pool"
    out["fold"] = -1
    out.loc[pool, "fold"] = _patient_folds(
        grades.loc[pool, "exp"].astype(int), grades.loc[pool, "patient"], N_FOLDS, seed
    )
    out["gold_patient"] = out.patient.isin(gold_patients)
    return out.drop(columns="label_set")
