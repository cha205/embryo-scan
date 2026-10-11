"""Rank embryos by viability from their first 3 days of division timings.

Outcome: did the embryo form a blastocyst (reach tB) by day 5 (116 h)? Only
videos recorded for at least 116 h count, so a video that simply stopped early
(e.g. a day-3 transfer) isn't mistaken for an embryo that stalled. Blastocyst
formation is the standard viability proxy in time-lapse research; embryos
that stall can't be transferred at day 5.

Features use only what's known by day 3 (72 h): t2 ... t8 and the derived
cc2 (2nd cell-cycle length), s2 (2nd division synchrony) and direct cleavage.
Models: the published Meseguer 2011 categories (no training) and a logistic
regression trained on train+val videos. Both are tested on the held-out
patients' videos twice: with the embryologist's annotated timings, and end to
end with the timings our model detected from the video frames (day 0-3 only).

Usage: python scripts/viability.py [--run r34]
"""

import argparse
import json
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from embryoscan import paths
from embryoscan.metrics import auc_ci
from embryoscan.report import ci, md_table
from embryoscan.stages import load_table, make_model, predict_frames
from embryoscan.timelapse import (CATEGORY_ORDER, P, decode_monotonic, direct_cleavage, meseguer_category,
                                  morphokinetics, onsets)

DAY3, DAY5 = 72.0, 116.0
EARLY = ["t2", "t3", "t4", "t5", "t6", "t7", "t8"]
FEATURES = EARLY + ["cc2", "s2", "direct", "reached_t8"]


def features_from_onsets(on: dict) -> dict:
    on = {k: v for k, v in on.items() if v <= DAY3}  # only what is known by day 3
    m = morphokinetics(on)
    row = {k: on.get(k, np.nan) for k in EARLY}
    row.update(cc2=m["cc2"], s2=m["s2"], direct=float(direct_cleavage(m)), reached_t8=float("t8" in on))
    row["category"] = meseguer_category(m)
    return row


def outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for video, g in df.groupby("video", sort=False):
        h, ph = g["hours"].to_numpy(), g["phase"].to_numpy()
        if h.max() < DAY5:
            continue
        r = features_from_onsets(onsets(ph, h))
        r.update(video=video, patient=g.patient.iloc[0], split=g.split.iloc[0],
                 blastocyst=int(((ph >= P["tB"]) & (h <= DAY5)).any()))
        rows.append(r)
    return pd.DataFrame(rows)


def model():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def category_score(cats: pd.Series) -> np.ndarray:
    """Higher = better: A+ -> 8 ... D- -> 1, X (couldn't be scored) -> 0."""
    return cats.map(lambda c: len(CATEGORY_ORDER) - 1 - CATEGORY_ORDER.index(c)).to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r34")
    ap.add_argument("--arch", default="resnet34.a1_in1k")
    a = ap.parse_args()
    df = load_table()
    tab = outcome_table(df)
    train, test = tab[tab.split != "test"], tab[tab.split == "test"]
    lr = model().fit(train[FEATURES].to_numpy(float), train.blastocyst)

    # end to end: timings detected by our model from frames up to day 3
    net = make_model(a.arch, pretrained=False).cuda()
    net.load_state_dict(torch.load(paths.RUNS / "timelapse" / a.run / "model.pt", map_location="cuda"))
    net = net.to(memory_format=torch.channels_last).eval()
    ai_rows = []
    for video in test.video:
        g = df[(df.video == video) & (df.hours <= DAY3)]
        dec = decode_monotonic(predict_frames(net, g["row"].to_numpy(), tta=True))
        r = features_from_onsets(onsets(dec, g["hours"].to_numpy()))
        r["video"] = video
        ai_rows.append(r)
    ai = pd.DataFrame(ai_rows).merge(test[["video", "patient", "blastocyst"]], on="video")

    y, groups = test.blastocyst.to_numpy(), test.patient.to_numpy()
    res, rows = {}, []
    for name, frame in (("embryologist's timings", test), ("AI-detected timings (end to end)", ai)):
        for method, score in (("Meseguer 2011 categories", category_score(frame["category"])),
                              ("logistic regression on timings", lr.predict_proba(frame[FEATURES].to_numpy(float))[:, 1])):
            auc, lo, hi = auc_ci(frame.blastocyst.to_numpy(), score, frame.patient.to_numpy())
            res[f"{name} | {method}"] = {"auc": auc, "lo": lo, "hi": hi}
            rows.append([name, method, ci(auc, lo, hi)])
    by_cat = test.groupby("category").blastocyst.agg(["count", "mean"]).reindex(CATEGORY_ORDER).dropna().reset_index()
    by_cat.columns = ["Meseguer category", "videos", "blastocyst by day 5"]

    out = paths.RUNS / "timelapse" / a.run
    ai["score"] = lr.predict_proba(ai[FEATURES].to_numpy(float))[:, 1]
    ai.to_csv(out / "viability_test_ai.csv", index=False)
    with open(out / "viability_lr.pkl", "wb") as f:
        pickle.dump({"model": lr, "features": FEATURES}, f)
    (out / "viability.json").write_text(json.dumps(res, indent=2))
    text = ["# Viability ranking from day 0-3 division timings\n",
            f"Outcome: blastocyst formed by day 5 ({DAY5:.0f} h). Videos recorded at least that long: "
            f"{len(tab)} ({tab.blastocyst.mean():.0%} formed a blastocyst). Logistic regression trained on "
            f"{len(train)} train/val videos; tested on {len(test)} videos from held-out patients "
            f"({int(y.sum())} blastocysts). AUC with 95% patient-bootstrap CI.\n",
            md_table(pd.DataFrame(rows, columns=["timings from", "ranking method", "AUC (95% CI)"])),
            "\nBlastocyst rate by Meseguer category (embryologist's timings, test videos):\n",
            md_table(by_cat, "{:.2f}")]
    (paths.REPORTS / "viability.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
