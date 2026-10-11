"""Patient-level data splits.

Every image from one patient lands in the same split. Sibling embryos share a
patient's genetics, age, clinic day and microscope settings, so splitting by
image would let a model "recognise the patient" and inflate its scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 2026
N_FOLDS = 5


def _patient_folds(labels: pd.Series, patients: pd.Series, n_splits: int, seed: int) -> np.ndarray:
    """Assign each row a fold so that patients never straddle folds and the
    label rate is similar in every fold."""
    folds = np.full(len(labels), -1)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(cv.split(np.zeros(len(labels)), labels.to_numpy(), patients.to_numpy())):
        folds[idx] = k
    return folds


def make_outcome_split(clinical: pd.DataFrame, target: str = "lb", seed: int = SEED) -> pd.DataFrame:
    """Lock away ~20% of patients as the final test set, and split the rest into
    5 cross-validation folds for model development.

    Returns: file, patient, split ('dev' | 'test'), fold (0-4 for dev, -1 for test).
    """
    first = _patient_folds(clinical[target], clinical["patient"], N_FOLDS, seed)
    out = clinical[["file", "patient"]].copy()
    out["split"] = np.where(first == 0, "test", "dev")
    dev = out["split"] == "dev"
    out["fold"] = -1
    out.loc[dev, "fold"] = _patient_folds(clinical.loc[dev, target], clinical.loc[dev, "patient"], N_FOLDS, seed + 1)
    return out


def make_grade_split(grades: pd.DataFrame, outcome_split: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Split the grade-labelled images for training the grading model.

    split values:
      pool       silver-labelled training images, in 5 patient-level folds. One
                 grading model is trained per fold ("cross-fitting"), so every
                 pool image gets a grade from a model that never saw it.
      gold_test  the 300 multi-expert images, used only for the grading benchmark
      held_out   every image of a patient in the outcome test set, so the final
                 outcome test never contains an image any model was trained on
      unused     the image with conflicting labels and the unlabelled one
    gold_patient marks patients who also have a gold image (for a stricter,
    patient-disjoint grading check).
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
