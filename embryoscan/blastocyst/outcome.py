"""Outcome (live birth) prediction: features, models and cross-validation.

All preprocessing (imputation, scaling) is fitted inside each training fold,
never on the fold being predicted.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import paths
from .grading import DEFAULT_RUN

CLINICAL = ["age", "log_amh", "endo", "coc", "mii", "day5"]
EXPERT_GRADE = ["exp", "icm_a", "icm_b", "icm_c", "te_a", "te_b", "te_c"]
AI_GRADE = [f"p_exp{c}" for c in range(5)] + [f"p_icm{c}" for c in range(3)] + [f"p_te{c}" for c in range(3)]
TARGETS = {"lb": "live-birth", "ha": "heartbeat", "ss": "pregnancy-test"}  # column -> file-name label


def add_image_features(df: pd.DataFrame, run: str = DEFAULT_RUN) -> tuple[pd.DataFrame, list[str]]:
    """Join the grading model's grade probabilities and pooled image features.
    Every image's values come from models that never trained on it (out-of-fold
    for training images, the 5-model average for held-out ones)."""
    d = paths.run_dir("blastocyst", run)
    pred = pd.read_csv(d / "predictions.csv")
    feats = np.load(d / "features.npy")
    emb_cols = [f"emb{i}" for i in range(feats.shape[1])]
    emb = pd.DataFrame(feats, columns=emb_cols)
    emb["file"] = pred["file"].to_numpy()
    out = df.merge(pred[["file"] + AI_GRADE], on="file", how="left").merge(emb, on="file", how="left")
    missing = out[AI_GRADE[0]].isna().sum()
    if missing:
        raise ValueError(f"{missing} outcome images have no image features")
    return out, emb_cols


def embedding_model(n_emb: int, n_other: int, components: int = 16, C: float = 0.1) -> Callable:
    """PCA-compress the 768-d image embedding (fitted per training fold), then
    logistic regression together with any other (clinical) columns."""

    def make():
        steps = [("emb", make_pipeline(StandardScaler(), PCA(components, random_state=0)), list(range(n_emb)))]
        if n_other:
            steps.append(
                (
                    "other",
                    make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
                    list(range(n_emb, n_emb + n_other)),
                )
            )
        return make_pipeline(ColumnTransformer(steps), StandardScaler(), LogisticRegression(C=C, max_iter=3000))

    return make


def load_outcome_table() -> pd.DataFrame:
    """Clinical rows joined with the patient-level outcome split and derived features."""
    c = pd.read_csv(paths.BLASTOCYST_CLINICAL)
    s = pd.read_csv(paths.SPLIT_OUTCOME)
    df = c.merge(s[["file", "split", "fold"]], on="file", validate="one_to_one")
    df["log_amh"] = np.log(df["amh"])
    df["log_amh_lo"] = np.log(df["amh_lo"])
    df["log_amh_hi"] = np.log(df["amh_hi"])
    df["day5"] = (df["day"] == 5).astype(float)
    df["exp"] = df["exp_silver"].astype(float)
    for col, src in (("icm", "icm_silver"), ("te", "te_silver")):
        for code, letter in enumerate("abc"):
            df[f"{col}_{letter}"] = (df[src] == code).astype(float)
    return df


def logistic(C: float = 0.5) -> Callable:
    def make():
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=C, max_iter=2000),
        )

    return make


def lightgbm() -> Callable:
    def make():
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.02,
            num_leaves=7,
            min_child_samples=25,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            verbose=-1,
            random_state=0,
        )

    return make


def cv_predict(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, make_model: Callable) -> np.ndarray:
    """Out-of-fold probabilities over the dev set's 5 patient-level folds."""
    oof = np.full(len(df), np.nan)
    for k in sorted(df["fold"].unique()):
        tr, te = (df["fold"] != k).to_numpy(), (df["fold"] == k).to_numpy()
        model = make_model()
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return oof
