"""Sanity check: score the paper's own XCeption predictions (shipped with the
dataset) with embryoscan.metrics.paper_grading_metrics. If the numbers match
Table 2 of Kromp et al. 2023 (expansion 0.78, ICM 0.69, TE 0.62), our grading
evaluation is computed the same way as the published benchmark.
"""

import json

import pandas as pd

from embryoscan import paths
from embryoscan.data import image_index, image_key, load_grades
from embryoscan.metrics import paper_grading_metrics

PAPER_TABLE2 = {"exp": 0.78, "icm": 0.69, "te": 0.62}


def main():
    lookup = {(p, e): f for f, p, e in image_index()[["file", "patient", "embryo"]].itertuples(index=False)}
    pred = pd.read_csv(paths.RAW / "prediction_xception.csv", header=None, names=["Image", "exp", "icm", "te"])
    pred["file"] = pred["Image"].map(lambda n: lookup[image_key(n)])
    gold = load_grades().query("label_set == 'gold'")[["file", "exp", "icm", "te"]]
    res = paper_grading_metrics(pred[["file", "exp", "icm", "te"]], gold)
    print(json.dumps(res, indent=2))
    for k, v in PAPER_TABLE2.items():
        ok = abs(res[k]["accuracy"] - v) < 0.006
        print(f"{k}: ours {res[k]['accuracy']:.3f} vs paper {v:.2f} -> {'MATCH' if ok else 'MISMATCH'}")
    paths.REPORTS.mkdir(exist_ok=True)
    (paths.REPORTS / "paper_xception_check.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
