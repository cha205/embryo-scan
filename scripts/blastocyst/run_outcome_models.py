"""Compare image-based live-birth models with the baselines, using patient-level 5-fold CV on the dev set.

Image inputs come from the grading model: its 11 grade probabilities ("automated grade")
and its 768-d image embedding compressed with PCA ("automated image embedding").

Usage: python scripts/blastocyst/run_outcome_models.py [--target lb] [--run grader-convnext-tiny]
"""

import argparse
import json

import pandas as pd

from embryoscan import paths
from embryoscan.blastocyst.grading import DEFAULT_RUN
from embryoscan.blastocyst.outcome import (
    AI_GRADE,
    CLINICAL,
    EXPERT_GRADE,
    TARGETS,
    add_image_features,
    cv_predict,
    embedding_model,
    load_outcome_table,
    logistic,
)
from embryoscan.common.metrics import auc_diff_ci, binary_summary
from embryoscan.common.reporting import ci, md_table


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--target", default="lb", choices=list(TARGETS), help="lb live birth, ha heartbeat, ss pregnancy test"
    )
    ap.add_argument("--run", default=DEFAULT_RUN, help="grader run folder under runs/blastocyst/")
    a = ap.parse_args()
    label = TARGETS[a.target]

    df, emb_cols = add_image_features(load_outcome_table(), a.run)
    dev = df[df.split == "dev"].reset_index(drop=True)
    y = dev[a.target].to_numpy()
    groups = dev["patient"].to_numpy()
    E = len(emb_cols)

    models = {
        "age only": (["age"], logistic()),
        "embryologist's grade": (EXPERT_GRADE + ["day5"], logistic()),
        "automated grade": (AI_GRADE + ["day5"], logistic()),
        "automated image embedding": (emb_cols, embedding_model(E, 0)),
        "automated grade + clinical": (AI_GRADE + CLINICAL, logistic()),
        "automated embedding + clinical": (emb_cols + CLINICAL, embedding_model(E, len(CLINICAL))),
        "automated grade + embryologist's grade + age": (AI_GRADE + EXPERT_GRADE + ["age", "day5"], logistic()),
    }
    oof = pd.DataFrame({"file": dev["file"], "patient": dev["patient"], "fold": dev["fold"], "y": y})
    res = {}
    for name, (cols, make) in models.items():
        oof[name] = cv_predict(dev, dev[cols].to_numpy(float), y, make)
        res[name] = binary_summary(y, oof[name], groups)

    rows = []
    for name, s in res.items():
        cmp = []
        for base in ("age only", "embryologist's grade"):
            if name == base:
                cmp.append("")
                continue
            d, lo, hi, p = auc_diff_ci(y, oof[name], oof[base], groups)
            s[f"vs {base}"] = {"diff": d, "lo": lo, "hi": hi, "p_not_better": p}
            cmp.append(f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})")
        rows.append([name, ci(s["auc"], s["auc_lo"], s["auc_hi"]), *cmp, f"{s['brier']:.3f}"])

    out = paths.run_dir("blastocyst", "outcome")
    out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out / f"oof-{label}-image-models.csv", index=False)
    (out / f"summary-{label}-image-models.json").write_text(json.dumps(res, indent=2))
    text = [
        f"# Image-based models compared with baselines: {label.replace('-', ' ')}\n",
        f"The development set comprises {len(dev)} embryos from {dev.patient.nunique()} patients, with "
        f"{int(y.sum())} positive outcomes. Models were evaluated by patient-level five-fold cross-validation, with "
        "95% confidence intervals from a patient-level bootstrap; differences between models are paired.\n",
        md_table(
            pd.DataFrame(
                rows,
                columns=[
                    "model",
                    "AUC (95% CI)",
                    "difference vs. age only",
                    "difference vs. embryologist's grade",
                    "Brier score",
                ],
            )
        ),
    ]
    paths.report("blastocyst", f"outcome-{label}.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
