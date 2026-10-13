"""Check whether patients shared between the grading train and test sets inflate accuracy.

The gold test set was drawn by image, so 30% of training images come from
patients who also have a test image. On the gold set, this compares the reported
5-fold ensemble with one trained without those patients and with a size-matched
control that drops the same number of random images. Writes
reports/blastocyst/patient-overlap-check.md.

Usage: python scripts/blastocyst/check_patient_overlap.py
"""

import argparse

import numpy as np
import pandas as pd

from embryoscan import paths
from embryoscan.blastocyst.grading import DEFAULT_RUN
from embryoscan.common.metrics import cluster_resamples, paper_grading_metrics
from embryoscan.common.reporting import md_table

COLS = ["file", "exp", "icm", "te"]
CRIT = ("exp", "icm", "te")
COLUMN_NAMES = ["expansion", "inner cell mass", "trophectoderm"]
WHOLE, STRICT, CONTROL = "whole pool (as reported)", "without gold patients", "size-matched control"
RUNS = {WHOLE: DEFAULT_RUN, STRICT: "grader-no-gold-patients", CONTROL: "grader-size-matched"}


def bootstrap(preds: dict[str, pd.DataFrame], gold: pd.DataFrame, n_boot: int) -> dict[str, dict[str, np.ndarray]]:
    """Accuracy of every run on the same patient-level resamples of the gold set."""
    aligned = {name: gold[["file"]].merge(p[COLS], on="file", how="left") for name, p in preds.items()}
    out = {name: {c: [] for c in CRIT} for name in preds}
    for idx in cluster_resamples(gold["patient"].to_numpy(), n_boot, seed=0):
        truth = gold.iloc[idx][COLS].drop_duplicates("file")
        for name, p in aligned.items():
            m = paper_grading_metrics(p.iloc[idx], truth)
            for c in CRIT:
                out[name][c].append(m[c]["accuracy"])
    return {name: {c: np.asarray(v) for c, v in d.items()} for name, d in out.items()}


def ci(values: np.ndarray, signed: bool = False) -> str:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return f"{lo:+.3f} to {hi:+.3f}" if signed else f"{lo:.2f}-{hi:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    a = ap.parse_args()
    grades = pd.read_csv(paths.BLASTOCYST_GRADES)
    pool = pd.read_csv(paths.SPLIT_GRADING).query("split == 'pool'")
    gold = grades[grades.label_set == "gold"][["file", "patient", "exp", "icm", "te"]].reset_index(drop=True)
    preds = {name: pd.read_csv(paths.run_dir("blastocyst", run) / "predictions.csv") for name, run in RUNS.items()}
    acc = {name: paper_grading_metrics(p[COLS], gold[COLS]) for name, p in preds.items()}
    boots = bootstrap(preds, gold, a.boot)

    whole = np.mean([(pool.fold != k).sum() for k in range(5)])
    reduced = np.mean([((pool.fold != k) & ~pool.gold_patient).sum() for k in range(5)])
    size = {WHOLE: whole, STRICT: reduced, CONTROL: reduced}
    rows = [
        [name, f"{size[name]:,.0f}", *[f"{acc[name][c]['accuracy']:.3f} ({ci(boots[name][c])})" for c in CRIT]]
        for name in RUNS
    ]
    contrasts = [
        ("without gold patients minus size-matched control (overlap effect)", STRICT, CONTROL),
        ("size-matched control minus whole pool (effect of reduced training data)", CONTROL, WHOLE),
    ]
    diffs = [
        [label, *[f"{acc[x][c]['accuracy'] - acc[y][c]['accuracy']:+.3f} ({ci(boots[x][c] - boots[y][c], True)})"
                  for c in CRIT]]
        for label, x, y in contrasts
    ]  # fmt: skip

    seen = gold.patient.isin(pool.patient)
    by_patient = []
    for name, p in preds.items():
        row = [name]
        for subset in (seen, ~seen):
            m = paper_grading_metrics(p[COLS], gold.loc[subset, COLS])
            row.append(" / ".join(f"{m[c]['accuracy']:.3f}" for c in CRIT))
        by_patient.append(row)

    text = [
        "# Effect of patient overlap on the grading results\n",
        f"The gold test set was drawn by image rather than by patient: {int(pool.gold_patient.sum())} of "
        f"{len(pool):,} training-pool images ({pool.gold_patient.mean():.0%}) come from "
        f"{pool[pool.gold_patient].patient.nunique()} patients who also contribute a gold test image. Because sibling "
        "embryos resemble one another, a model could achieve higher accuracy on patients represented in its training "
        "data. Three five-fold ensembles with identical settings were therefore scored on the 300 gold images "
        f"(accuracy with 95% patient-level bootstrap confidence intervals, {a.boot:,} resamples):\n",
        md_table(pd.DataFrame(rows, columns=["model", "training images per fold", *COLUMN_NAMES])),
        "\nPaired differences were computed on identical resamples (95% CI). If the overlap inflated the results, "
        "the differences in the first row would be clearly negative:\n",
        md_table(pd.DataFrame(diffs, columns=["comparison", *COLUMN_NAMES])),
        "\nAccuracy (expansion / ICM / TE) on gold images from patients represented in the training pool "
        f"({int(seen.sum())} images) and from patients absent from it ({int((~seen).sum())} images):\n",
        md_table(
            pd.DataFrame(by_patient, columns=["model", "patient in the training pool", "patient not in the pool"])
        ),
        "\nThe intervals reflect the sampling of embryos in the test set, not the randomness of training; the "
        "single models of the reported run differ by up to 0.04 in expansion accuracy from one fold to the next.",
    ]
    paths.report("blastocyst", "patient-overlap-check.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
