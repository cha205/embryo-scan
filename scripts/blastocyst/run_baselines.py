"""Non-image baselines for live birth, scored with patient-level 5-fold
cross-validation on the dev set (the locked test set is not touched).

Usage: python scripts/blastocyst/run_baselines.py [--target lb|ha|ss]
"""

import argparse
import json

import pandas as pd

from embryoscan import paths
from embryoscan.blastocyst.outcome import (
    CLINICAL,
    EXPERT_GRADE,
    TARGETS,
    cv_predict,
    lightgbm,
    load_outcome_table,
    logistic,
)
from embryoscan.common.metrics import auc_diff_ci, binary_summary
from embryoscan.common.reporting import ci, md_table

MODELS = {
    "age only": (["age"], logistic()),
    "embryologist's grade": (EXPERT_GRADE + ["day5"], logistic()),
    "all clinical data": (CLINICAL, logistic()),
    "all clinical data (LightGBM)": (CLINICAL, lightgbm()),
    "embryologist's grade + clinical": (EXPERT_GRADE + CLINICAL, logistic()),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--target", default="lb", choices=list(TARGETS), help="lb live birth, ha heartbeat, ss pregnancy test"
    )
    a = ap.parse_args()
    label = TARGETS[a.target]
    df = load_outcome_table()
    dev = df[df.split == "dev"].reset_index(drop=True)
    y = dev[a.target].to_numpy()
    groups = dev["patient"].to_numpy()

    oof = pd.DataFrame({"file": dev["file"], "patient": dev["patient"], "fold": dev["fold"], "y": y})
    rows, results = [], {}
    for name, (cols, make) in MODELS.items():
        p = cv_predict(dev, dev[cols].to_numpy(float), y, make)
        oof[name] = p
        s = binary_summary(y, p, groups)
        results[name] = s
    for name, s in results.items():
        if name == "age only":
            vs = ""
        else:
            d, lo, hi, p_nb = auc_diff_ci(y, oof[name], oof["age only"], groups)
            vs = f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})"
            s["vs_age"] = {"diff": d, "lo": lo, "hi": hi, "p_not_better": p_nb}
        rows.append([name, ci(s["auc"], s["auc_lo"], s["auc_hi"]), vs, f"{s['brier']:.3f}"])

    # Sensitivity: AMH decoded with the low vs high reading of ambiguous dates
    sens = {}
    for which in ("lo", "hi"):
        cols = ["age", f"log_amh_{which}", "endo", "coc", "mii", "day5"]
        sens[which] = binary_summary(y, cv_predict(dev, dev[cols].to_numpy(float), y, logistic()), groups)["auc"]

    out_dir = paths.run_dir("blastocyst", "baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_dir / f"oof-{label}.csv", index=False)
    (out_dir / f"summary-{label}.json").write_text(json.dumps({"models": results, "amh_sensitivity": sens}, indent=2))

    names = {"lb": "live birth", "ha": "ongoing pregnancy with heartbeat", "ss": "positive pregnancy test"}
    text = [
        f"# Baseline models: {names[a.target]}\n",
        f"The development set comprises {len(dev)} embryos from {dev.patient.nunique()} patients, with "
        f"{int(y.sum())} positive outcomes ({y.mean():.0%}). Models were evaluated by patient-level five-fold "
        "cross-validation, with 95% confidence intervals from a patient-level bootstrap, and are logistic "
        "regressions unless stated otherwise.\n",
        md_table(pd.DataFrame(rows, columns=["model", "AUC (95% CI)", "AUC difference vs. age only", "Brier score"])),
        "\nIn a sensitivity analysis in which ambiguous AMH values were read as the lower or the higher candidate "
        f"value, the AUC of the clinical model was {sens['lo']:.3f} and {sens['hi']:.3f}, respectively.\n",
        "An AUC of 0.5 corresponds to chance-level discrimination. The Brier score is the mean squared error of the "
        "predicted probability (lower is better); a constant prediction of the base rate scores "
        f"{y.mean() * (1 - y.mean()):.3f}.",
    ]
    paths.report("blastocyst", f"baselines-{label}.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
