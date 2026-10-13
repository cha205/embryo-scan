"""Compare the grading model with individual experts and published models on the gold test set.

Writes reports/blastocyst/grading.md and reports/figures/blastocyst-grading-vs-experts.png.

Usage: python scripts/blastocyst/report_grading.py [--run grader-convnext-tiny]
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from embryoscan import paths
from embryoscan.blastocyst.grading import DEFAULT_RUN
from embryoscan.common import figures
from embryoscan.common.figures import plt
from embryoscan.common.metrics import cluster_resamples, paper_grading_metrics
from embryoscan.common.reporting import md_table

COLS = ["file", "exp", "icm", "te"]
CRIT = ("exp", "icm", "te")
# Kromp et al. 2023, Table 2: individual embryologists against the panel, and the best published model
EXPERTS = {"exp": 0.78, "icm": 0.74, "te": 0.70}
PUBLISHED = {"exp": 0.85, "icm": 0.69, "te": 0.62}


def accuracy_ci(pred, gold, n_boot=2000):
    """Patient-level bootstrap CI for each criterion's accuracy."""
    m = gold.merge(pred, on="file", suffixes=("_gt", ""))
    boots = {c: [] for c in CRIT}
    for idx in cluster_resamples(m["patient"].to_numpy(), n_boot, seed=0):
        r = paper_grading_metrics(
            m.iloc[idx][COLS],
            m.iloc[idx][["file", "exp_gt", "icm_gt", "te_gt"]].set_axis(COLS, axis=1).drop_duplicates("file"),
        )
        for c in CRIT:
            boots[c].append(r[c]["accuracy"])
    return {c: np.percentile(boots[c], [2.5, 97.5]) for c in CRIT}


def confusion(pred, gold, crit):
    m = gold.merge(pred, on="file", suffixes=("_gt", ""))
    names = {
        "exp": {i: str(i + 1) for i in range(5)},
        "icm": {0: "A", 1: "B", 2: "C", 3: "n/d"},
        "te": {0: "A", 1: "B", 2: "C", 3: "n/d"},
    }[crit]
    gt = m[f"{crit}_gt"].map(lambda v: names.get(int(v), "n/a") if v == v else "n/a")
    pr = m[crit].map(lambda v: names[int(v)])
    t = pd.crosstab(gt, pr).reindex(index=[*names.values(), "n/a"], columns=list(names.values()), fill_value=0)
    t = t.loc[t.sum(axis=1) > 0]
    t.index.name = "panel \\ model"
    return md_table(t.reset_index(), "{:.0f}")


def plot_vs_experts(ens: dict, cis: dict) -> None:
    """Accuracy per criterion: individual embryologists, the best published
    model and the ensemble (with its 95% CI)."""
    groups = [
        ("Individual embryologists (mean)", [EXPERTS[c] for c in CRIT], figures.FAINT, figures.INK),
        ("Best published model (Kromp et al., 2023)", [PUBLISHED[c] for c in CRIT], figures.MUTED, "white"),
        ("Embryo Scan ensemble", [ens[c]["accuracy"] for c in CRIT], figures.COBALT, "white"),
    ]
    x, width = np.arange(len(CRIT)), 0.27
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, (name, values, color, text) in enumerate(groups):
        bars = ax.bar(x + (i - 1) * width, values, width * 0.9, color=color, label=name)
        ax.bar_label(bars, fmt="%.2f", label_type="center", color=text, fontsize=9.5, fontweight="semibold")
    ours = np.array(groups[-1][1])
    ci = np.array([cis[c] for c in CRIT]).T
    ax.errorbar(x + width, ours, yerr=[ours - ci[0], ci[1] - ours], fmt="none", ecolor=figures.INK, capsize=4, lw=1)
    ax.set_xticks(x, ["Expansion", "Inner cell mass", "Trophectoderm"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy relative to the expert panel")
    ax.set_title("Gardner grading accuracy on the 300-image consensus test set")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    figures.save(fig, "blastocyst-grading-vs-experts")


BIAS_GRID = np.linspace(0, 3, 31)


def tune_bias(p: np.ndarray, y: np.ndarray, grid: np.ndarray = BIAS_GRID) -> np.ndarray:
    """Additive log-probability bonus for classes B and C that maximizes
    balanced accuracy (mean recall over A/B/C). Fitted on out-of-fold
    predictions for the training pool, never on the gold test set."""
    logp = np.log(np.clip(p, 1e-6, 1))
    best, best_b = -1.0, np.zeros(3)
    for b1 in grid:
        for b2 in grid:
            b = np.array([0, b1, b2])
            score = balanced_accuracy_score(y, (logp + b).argmax(1))
            if score > best:
                best, best_b = score, b
    return best_b


def adjusted(pred: pd.DataFrame, bias: dict) -> pd.DataFrame:
    out = pred.copy()
    for k, b in bias.items():
        cols = [f"p_{k}{c}" for c in range(3)]
        out[k] = (np.log(np.clip(out[cols].to_numpy(), 1e-6, 1)) + b).argmax(1)
        out.loc[out["exp"] <= 1, k] = 3
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=DEFAULT_RUN, help="grader run folder under runs/blastocyst/")
    a = ap.parse_args()
    d = paths.run_dir("blastocyst", a.run)
    grades = pd.read_csv(paths.BLASTOCYST_GRADES)
    gold = grades[grades.label_set == "gold"][["file", "patient", "exp", "icm", "te"]]
    final = pd.read_csv(d / "predictions.csv")
    ens = paper_grading_metrics(final[COLS], gold[COLS])
    folds = [paper_grading_metrics(pd.read_csv(d / f"fold-{k}-predictions.csv")[COLS], gold[COLS]) for k in range(5)]
    cis = accuracy_ci(final[COLS], gold)
    paper_xc = json.loads(paths.report("blastocyst", "paper-check.json").read_text())

    def fold_range(c):
        v = [f[c]["accuracy"] for f in folds]
        return f"{np.mean(v):.3f} ({min(v):.3f}-{max(v):.3f})"

    plot_vs_experts(ens, cis)
    rows = [
        [
            "Individual embryologists vs. panel (Kromp et al., 2023; mean ± SD)",
            "0.78 ± 0.12",
            "0.74 ± 0.06",
            "0.70 ± 0.14",
        ],
        ["Best published model (Kromp et al., 2023)", "0.85 (Swin)", "0.69 (Xception)", "0.62 (Xception/Swin)"],
        ["Published Xception predictions, re-scored with this code", *[f"{paper_xc[c]['accuracy']:.3f}" for c in CRIT]],
        ["Embryo Scan, single models (mean, range)", *[fold_range(c) for c in CRIT]],
        [
            "**Embryo Scan, five-model ensemble** (95% CI)",
            *[f"**{ens[c]['accuracy']:.3f}** ({cis[c][0]:.2f}-{cis[c][1]:.2f})" for c in CRIT],
        ],
    ]
    # Class-balance correction tuned on out-of-fold training predictions
    pool = final[final.split == "pool"].merge(grades[["file", "exp", "icm", "te"]], on="file", suffixes=("", "_silver"))
    bias = {}
    for k in ("icm", "te"):
        ok = pool[f"{k}_silver"].isin([0, 1, 2])
        bias[k] = tune_bias(
            pool.loc[ok, [f"p_{k}{c}" for c in range(3)]].to_numpy(), pool.loc[ok, f"{k}_silver"].astype(int).to_numpy()
        )
    adj = adjusted(final, bias)
    ens_adj = paper_grading_metrics(adj[COLS], gold[COLS])

    def macro_f1(pred, k):
        mm = gold.merge(pred[COLS], on="file", suffixes=("_gt", ""))
        mm = mm[(mm.exp_gt >= 2) & (mm.exp >= 2) & mm[f"{k}_gt"].isin([0, 1, 2]) & mm[k].isin([0, 1, 2])]
        return (
            f1_score(mm[f"{k}_gt"], mm[k], average="macro"),
            (mm[mm[f"{k}_gt"] == 2][k] == 2).sum(),
            (mm[f"{k}_gt"] == 2).sum(),
        )

    adj_rows = []
    for name, pr, mt in (("argmax", final, ens), ("balanced thresholds", adj, ens_adj)):
        row = [name]
        for k in ("icm", "te"):
            f1m, c_hit, c_n = macro_f1(pr, k)
            row.append(f"{mt[k]['accuracy']:.3f} / {f1m:.3f} / {c_hit} of {c_n}")
        adj_rows.append(row)

    text = [
        "# Grading results\n",
        "The gold test set comprises 300 images graded by a panel of embryologists (majority vote), none of which "
        "was used to train a grading model. Accuracy is computed exactly as in the dataset authors' scoring script, "
        "in which the inner cell mass and the trophectoderm are scored only when neither the model nor the panel "
        "assigns expansion grade 1-2.\n",
        md_table(pd.DataFrame(rows, columns=["Grader", "Expansion", "Inner cell mass", "Trophectoderm"])),
        f"\nThe weighted F1 scores of the ensemble are {ens['exp']['f1_w']:.3f} for expansion, "
        f"{ens['icm']['f1_w']:.3f} for the inner cell mass and {ens['te']['f1_w']:.3f} for the trophectoderm, and "
        f"the quadratically weighted kappa for expansion is {ens['exp']['kappa_quadratic']:.3f}.\n",
        "The official test set was drawn by image rather than by patient, so 30% of the training images come from "
        "patients who also contribute a test image, as in the original benchmark. Retraining without those patients "
        "changes accuracy by less than 0.01 relative to a size-matched control "
        "([patient-overlap-check.md](patient-overlap-check.md)).\n",
        "## Balanced decision thresholds for rare grades\n",
        "Grade C is rare in the training data (16 inner cell mass and 47 trophectoderm labels among approximately "
        "1,700 images), and the argmax decision rule therefore seldom predicts it. Adding a class-specific offset to "
        "the log-probabilities of grades B and C, tuned to maximize balanced accuracy on the **out-of-fold training "
        "predictions** without use of the gold set, exchanges a small loss of overall accuracy for improved "
        "detection of rare grades. The offsets (log-probability, grades A/B/C) are "
        f"ICM {bias['icm'].round(2).tolist()} and TE {bias['te'].round(2).tolist()}.\n",
        md_table(
            pd.DataFrame(
                adj_rows,
                columns=[
                    "decision rule",
                    "ICM: accuracy / macro F1 / grade C detected",
                    "TE: accuracy / macro F1 / grade C detected",
                ],
            )
        ),
        "\n## Confusion matrices against the panel\n",
        "Expansion grade (rows: panel; columns: model):\n",
        confusion(final[COLS], gold, "exp"),
        "\nInner cell mass:\n",
        confusion(final[COLS], gold, "icm"),
        "\nTrophectoderm:\n",
        confusion(final[COLS], gold, "te"),
    ]
    paths.report("blastocyst", "grading.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
