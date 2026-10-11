"""Rank embryos by viability (blastocyst by day 5) from day 0-3 division timings.

Two ranking methods: the published Meseguer 2011 categories (no training) and a
logistic regression trained on train+val videos. Both are tested on held-out
patients' videos twice: with the embryologist's annotated timings, and end to
end with the timings the stage model detected from the frames (day 0-3 only).
Writes reports/timelapse/viability.md and reports/figures/timelapse-viability-roc.png.

Usage: python scripts/timelapse/run_viability.py [--run stages-resnet34]
"""

import argparse
import json
import pickle

import pandas as pd
import torch
from sklearn.metrics import roc_curve

from embryoscan import paths
from embryoscan.common import figures
from embryoscan.common.figures import plt
from embryoscan.common.metrics import auc_ci
from embryoscan.common.reporting import ci, md_table
from embryoscan.timelapse.labels import CATEGORY_ORDER, decode_monotonic, onsets
from embryoscan.timelapse.stages import DEFAULT_RUN, StageConfig, load_table, make_model, predict_frames
from embryoscan.timelapse.viability import DAY3, DAY5, FEATURES, category_score, features_from_onsets, outcome_table
from embryoscan.timelapse.viability import make_model as make_viability_model

SHORT = {
    "embryologist's timings": "embryologist's timings",
    "AI-detected timings (end to end)": "AI-detected timings",
    "Meseguer 2011 categories": "published rules",
    "logistic regression on timings": "learned model",
}


def plot_roc(curves: list[tuple[str, str, object, object, float]]) -> None:
    """ROC curve of every ranking: ink for the embryologist's timings, cobalt
    for the AI's; solid for the learned model, dashed for the published rules."""
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=figures.FAINT, lw=1, ls=":", label="coin flip (AUC 0.50)")
    for timings, method, y, score, auc in curves:
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(
            fpr,
            tpr,
            color=figures.COBALT if timings.startswith("AI") else figures.INK,
            ls="-" if method.startswith("logistic") else "--",
            lw=1.8,
            label=f"{SHORT[timings]}, {SHORT[method]} (AUC {auc:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("stalled embryos ranked as viable (false positive rate)")
    ax.set_ylabel("blastocysts ranked as viable (true positive rate)")
    ax.set_title("Ranking embryos from day 0-3 division timings")
    ax.legend(loc="lower right", fontsize=8.5)
    figures.save(fig, "timelapse-viability-roc")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=DEFAULT_RUN, help="stage-model run folder under runs/timelapse/")
    ap.add_argument("--arch", default=StageConfig.arch)
    a = ap.parse_args()
    run = paths.run_dir("timelapse", a.run)
    frames = load_table()
    tab = outcome_table(frames)
    train, test = tab[tab.split != "test"], tab[tab.split == "test"]
    lr = make_viability_model().fit(train[FEATURES].to_numpy(float), train.blastocyst)

    # end to end: timings detected by the stage model from frames up to day 3
    net = make_model(a.arch, pretrained=False).cuda()
    net.load_state_dict(torch.load(run / "model.pt", map_location="cuda"))
    net = net.to(memory_format=torch.channels_last).eval()
    ai_rows = []
    for video in test.video:
        g = frames[(frames.video == video) & (frames.hours <= DAY3)]
        dec = decode_monotonic(predict_frames(net, g["row"].to_numpy(), tta=True))
        r = features_from_onsets(onsets(dec, g["hours"].to_numpy()))
        r["video"] = video
        ai_rows.append(r)
    ai = pd.DataFrame(ai_rows).merge(test[["video", "patient", "blastocyst"]], on="video")

    y = test.blastocyst.to_numpy()
    res, rows, curves = {}, [], []
    for name, frame in (("embryologist's timings", test), ("AI-detected timings (end to end)", ai)):
        for method, score in (
            ("Meseguer 2011 categories", category_score(frame["category"])),
            ("logistic regression on timings", lr.predict_proba(frame[FEATURES].to_numpy(float))[:, 1]),
        ):
            auc, lo, hi = auc_ci(frame.blastocyst.to_numpy(), score, frame.patient.to_numpy())
            res[f"{name} | {method}"] = {"auc": auc, "lo": lo, "hi": hi}
            rows.append([name, method, ci(auc, lo, hi)])
            curves.append((name, method, frame.blastocyst.to_numpy(), score, auc))
    plot_roc(curves)
    by_cat = test.groupby("category").blastocyst.agg(["count", "mean"]).reindex(CATEGORY_ORDER).dropna().reset_index()
    by_cat.columns = ["Meseguer category", "videos", "blastocyst by day 5"]
    by_cat["videos"] = by_cat["videos"].astype(int)  # reindex made the counts float

    ai["score"] = lr.predict_proba(ai[FEATURES].to_numpy(float))[:, 1]
    ai.to_csv(run / "viability-scores-test.csv", index=False)
    with open(run / "viability-model.pkl", "wb") as f:
        pickle.dump({"model": lr, "features": FEATURES}, f)
    (run / "viability-summary.json").write_text(json.dumps(res, indent=2))
    text = [
        "# Viability ranking from day 0-3 division timings\n",
        f"Outcome: blastocyst formed by day 5 ({DAY5:.0f} h). Videos recorded at least that long: "
        f"{len(tab)} ({tab.blastocyst.mean():.0%} formed a blastocyst). Logistic regression trained on "
        f"{len(train)} train/val videos; tested on {len(test)} videos from held-out patients "
        f"({int(y.sum())} blastocysts). AUC with 95% patient-bootstrap CI.\n",
        md_table(pd.DataFrame(rows, columns=["timings from", "ranking method", "AUC (95% CI)"])),
        "\nBlastocyst rate by Meseguer category (embryologist's timings, test videos):\n",
        md_table(by_cat, "{:.2f}"),
    ]
    paths.report("timelapse", "viability.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
