"""Write reports/grading.md: the grading model vs individual experts and the
paper's published models on the 300-image gold test set.

Usage: python scripts/report_grading.py [--run convnext_t]
"""

import argparse
import json

import numpy as np
import pandas as pd

from embryoscan import paths
from embryoscan.metrics import _cluster_resamples, paper_grading_metrics
from embryoscan.report import md_table

COLS = ["file", "exp", "icm", "te"]
CRIT = ("exp", "icm", "te")


def accuracy_ci(pred, gold, n_boot=2000):
    """Patient-level bootstrap CI for each criterion's accuracy."""
    m = gold.merge(pred, on="file", suffixes=("_gt", ""))
    boots = {c: [] for c in CRIT}
    for idx in _cluster_resamples(m["patient"].to_numpy(), n_boot, seed=0):
        r = paper_grading_metrics(m.iloc[idx][COLS], m.iloc[idx][["file", "exp_gt", "icm_gt", "te_gt"]].set_axis(COLS, axis=1)
                                  .drop_duplicates("file"))
        for c in CRIT:
            boots[c].append(r[c]["accuracy"])
    return {c: np.percentile(boots[c], [2.5, 97.5]) for c in CRIT}


def confusion(pred, gold, crit):
    m = gold.merge(pred, on="file", suffixes=("_gt", ""))
    names = {"exp": {i: str(i + 1) for i in range(5)}, "icm": {0: "A", 1: "B", 2: "C", 3: "n/d"}, "te": {0: "A", 1: "B", 2: "C", 3: "n/d"}}[crit]
    gt = m[f"{crit}_gt"].map(lambda v: names.get(int(v), "n/a") if v == v else "n/a")
    pr = m[crit].map(lambda v: names[int(v)])
    t = pd.crosstab(gt, pr).reindex(index=[*names.values(), "n/a"], columns=list(names.values()), fill_value=0)
    t = t.loc[t.sum(axis=1) > 0]
    t.index.name = "panel \\ model"
    return md_table(t.reset_index(), "{:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="convnext_t")
    a = ap.parse_args()
    d = paths.RUNS / "grader" / a.run
    grades = pd.read_csv(paths.PROCESSED / "grades.csv")
    gold = grades[grades.label_set == "gold"][["file", "patient", "exp", "icm", "te"]]
    final = pd.read_csv(d / "predictions.csv")
    ens = paper_grading_metrics(final[COLS], gold[COLS])
    folds = [paper_grading_metrics(pd.read_csv(d / f"fold{k}_pred.csv")[COLS], gold[COLS]) for k in range(5)]
    cis = accuracy_ci(final[COLS], gold)
    paper_xc = json.loads((paths.REPORTS / "paper_xception_check.json").read_text())

    def fold_range(c):
        v = [f[c]["accuracy"] for f in folds]
        return f"{np.mean(v):.3f} ({min(v):.3f}-{max(v):.3f})"

    rows = [
        ["Individual experts vs panel (paper, mean ± SD)", "0.78 ± 0.12", "0.74 ± 0.06", "0.70 ± 0.14"],
        ["Best published model (paper)", "0.85 Swin", "0.69 XCeption", "0.62 XCeption/Swin"],
        ["Paper's XCeption, scored by our code", *[f"{paper_xc[c]['accuracy']:.3f}" for c in CRIT]],
        ["Embryo Scan, single models (mean, range)", *[fold_range(c) for c in CRIT]],
        ["**Embryo Scan, 5-model ensemble** (95% CI)", *[f"**{ens[c]['accuracy']:.3f}** ({cis[c][0]:.2f}-{cis[c][1]:.2f})" for c in CRIT]],
    ]
    text = [
        "# Grading results\n",
        "Gold test set: 300 images graded by a panel of embryologists (majority vote). No grading model trained on "
        "these images. Accuracy is computed exactly like the dataset authors' script: inner cell mass and "
        "trophectoderm are only scored when neither the model nor the panel puts the embryo at expansion grade 1-2.\n",
        md_table(pd.DataFrame(rows, columns=["", "Expansion", "Inner cell mass", "Trophectoderm"])),
        f"\nEnsemble weighted F1: expansion {ens['exp']['f1_w']:.3f}, ICM {ens['icm']['f1_w']:.3f}, TE {ens['te']['f1_w']:.3f}. "
        f"Expansion quadratic-weighted kappa: {ens['exp']['kappa_quadratic']:.3f}.\n",
        "Caveat: the official test set was drawn by image, not by patient, so some training images come from the "
        "same patients as test images (same as the paper's setup). A patient-disjoint check is planned.\n",
        "## Where the model and the panel disagree\n",
        "Expansion grade:\n", confusion(final[COLS], gold, "exp"),
        "\nInner cell mass:\n", confusion(final[COLS], gold, "icm"),
        "\nTrophectoderm:\n", confusion(final[COLS], gold, "te"),
    ]
    (paths.REPORTS / "grading.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
