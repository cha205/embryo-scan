"""Evaluation metrics with honest uncertainty.

AUC (area under the ROC curve) is the chance that the model scores a random
positive case above a random negative one: 0.5 is a coin flip, 1.0 is perfect.
Confidence intervals come from a patient-level (cluster) bootstrap, because two
embryos from one double-transfer patient are not independent samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def cluster_resamples(groups: np.ndarray, n_boot: int, seed: int):
    """Bootstrap resamples of row indices, drawing whole groups (patients)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(inv))
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([order[starts[k] : ends[k]] for k in pick])


def auc_ci(y, p, groups=None, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """AUC with a percentile bootstrap confidence interval: (auc, lo, hi)."""
    y, p = np.asarray(y), np.asarray(p)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    boots = []
    for idx in cluster_resamples(groups, n_boot, seed):
        yy = y[idx]
        if yy.min() != yy.max():
            boots.append(roc_auc_score(yy, p[idx]))
    a = (1 - level) / 2
    lo, hi = np.percentile(boots, [100 * a, 100 * (1 - a)])
    return float(roc_auc_score(y, p)), float(lo), float(hi)


def auc_diff_ci(y, p_new, p_base, groups=None, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Paired bootstrap of AUC(p_new) - AUC(p_base).

    Returns (diff, lo, hi, p_not_better) where p_not_better is the share of
    resamples in which the new model did not beat the baseline.
    """
    y, p_new, p_base = map(np.asarray, (y, p_new, p_base))
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    diffs = []
    for idx in cluster_resamples(groups, n_boot, seed):
        yy = y[idx]
        if yy.min() != yy.max():
            diffs.append(roc_auc_score(yy, p_new[idx]) - roc_auc_score(yy, p_base[idx]))
    diffs = np.asarray(diffs)
    a = (1 - level) / 2
    lo, hi = np.percentile(diffs, [100 * a, 100 * (1 - a)])
    diff = roc_auc_score(y, p_new) - roc_auc_score(y, p_base)
    return float(diff), float(lo), float(hi), float((diffs <= 0).mean())


def binary_summary(y, p, groups=None, n_boot: int = 2000, seed: int = 0) -> dict:
    auc, lo, hi = auc_ci(y, p, groups, n_boot, seed)
    return {
        "auc": auc,
        "auc_lo": lo,
        "auc_hi": hi,
        "brier": float(brier_score_loss(y, p)),
        "n": int(len(y)),
        "positives": int(np.sum(y)),
    }


def paper_grading_metrics(pred: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Grade accuracy computed exactly like the dataset authors' script
    (calculate_model_metrics.py), so numbers compare with their Table 2.

    pred, gold: columns file, exp, icm, te with dataset codes; gold NaN means
    'not assessable'. ICM and TE are only scored when neither the prediction
    nor the consensus puts the embryo at expansion grade 1-2 (codes 0-1).
    """
    m = gold.merge(pred, on="file", suffixes=("_gt", ""))

    def exp_code(v):
        return int(v) if v == v and int(v) in (0, 1, 2, 3, 4) else 5

    def cell_code(v):
        return int(v) if v == v and int(v) in (0, 1, 2) else 3

    exp_gt, exp_pr = m["exp_gt"].map(exp_code), m["exp"].map(exp_code)
    keep = ~(exp_pr.isin([0, 1]) | exp_gt.isin([0, 1]))
    out = {}
    pairs = {
        "exp": (exp_gt, exp_pr),
        "icm": (m.loc[keep, "icm_gt"].map(cell_code), m.loc[keep, "icm"].map(cell_code)),
        "te": (m.loc[keep, "te_gt"].map(cell_code), m.loc[keep, "te"].map(cell_code)),
    }
    for crit, (gt, pr) in pairs.items():
        prec, rec, f1, _ = precision_recall_fscore_support(gt, pr, average="weighted", zero_division=0)
        out[crit] = {
            "accuracy": float(accuracy_score(gt, pr)),
            "precision_w": float(prec),
            "recall_w": float(rec),
            "f1_w": float(f1),
            "n": int(len(gt)),
        }
    out["exp"]["kappa_quadratic"] = float(cohen_kappa_score(exp_gt, exp_pr, weights="quadratic"))
    return out
