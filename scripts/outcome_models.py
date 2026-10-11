"""Does the embryo photo help predict live birth? Image-based models vs the
baselines, with the same patient-level 5-fold CV on the dev set.

Image inputs come from the grading model (scripts/train_grader.py):
  AI grade      its 11 grade probabilities (expansion 1-5, ICM A-C, TE A-C)
  AI embedding  its 768-number internal description of the photo, PCA-compressed

Usage: python scripts/outcome_models.py [--target lb] [--run convnext_t]
"""

import argparse
import json

import numpy as np
import pandas as pd

from embryoscan import paths
from embryoscan.metrics import auc_diff_ci, binary_summary
from embryoscan.outcome import (AI_GRADE, CLINICAL, EXPERT_GRADE, add_image_features, cv_predict, embedding_model,
                                load_outcome_table, logistic)
from embryoscan.report import ci, md_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="lb", choices=["lb", "ha", "ss"])
    ap.add_argument("--run", default="convnext_t")
    a = ap.parse_args()

    df, emb_cols = add_image_features(load_outcome_table(), a.run)
    dev = df[df.split == "dev"].reset_index(drop=True)
    y = dev[a.target].to_numpy()
    groups = dev["patient"].to_numpy()
    E = len(emb_cols)

    models = {
        "age only": (["age"], logistic()),
        "embryologist's grade": (EXPERT_GRADE + ["day5"], logistic()),
        "AI grade": (AI_GRADE + ["day5"], logistic()),
        "AI embedding": (emb_cols, embedding_model(E, 0)),
        "AI grade + clinical": (AI_GRADE + CLINICAL, logistic()),
        "AI embedding + clinical": (emb_cols + CLINICAL, embedding_model(E, len(CLINICAL))),
        "AI grade + embryologist's grade + age": (AI_GRADE + EXPERT_GRADE + ["age", "day5"], logistic()),
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

    out = paths.RUNS / "outcome"
    out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out / f"oof_{a.target}_{a.run}.csv", index=False)
    (out / f"summary_{a.target}_{a.run}.json").write_text(json.dumps(res, indent=2))
    text = [f"# Image models vs baselines: {a.target}\n",
            f"Dev set: {len(dev)} embryos, {dev.patient.nunique()} patients, {int(y.sum())} positives. "
            "Patient-level 5-fold CV, 95% CIs from a patient-level bootstrap; differences are paired.\n",
            md_table(pd.DataFrame(rows, columns=["model", "AUC (95% CI)", "vs age only", "vs embryologist's grade", "Brier"]))]
    (paths.REPORTS / f"outcome_{a.target}.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
