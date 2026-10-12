"""Plot division timing from the stage model's saved predictions.

Writes reports/figures/timelapse-stage-timeline.png (one held-out embryo's stage
over five days, labelled and predicted) and timelapse-timing-errors.png (the
timing error of every division across the split).

Usage: python scripts/timelapse/plot_divisions.py [--run stages-resnet34] [--split test]
"""

import argparse

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from embryoscan import paths
from embryoscan.common import figures
from embryoscan.common.figures import plt
from embryoscan.timelapse.labels import DIVISIONS, PHASE_NAMES, P, timing_errors
from embryoscan.timelapse.stages import DEFAULT_RUN

Y_RANGE_H = 6  # the error axis shows +-6 hours


def typical_video(pred: pd.DataFrame, errors: pd.DataFrame) -> str:
    """The video whose median timing error is the median one, among videos
    the embryologist followed to the blastocyst stage."""
    reached = pred.groupby("video").phase.max() >= P["tB"]
    per_video = errors.assign(abs_h=errors.error_h.abs()).groupby("video").abs_h.median()
    per_video = per_video[per_video.index.isin(reached[reached].index)]
    return (per_video - per_video.median()).abs().idxmin()


def plot_timeline(pred: pd.DataFrame, video: str) -> None:
    g = pred[pred.video == video]
    labelled = g[g.phase >= 0]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.scatter(g.hours, g.raw, s=7, color=figures.COBALT, alpha=0.3, lw=0, label="model, each frame on its own")
    ax.step(g.hours, g.decoded, where="post", color=figures.COBALT, lw=2.2, label="model after decoding")
    ax.step(
        labelled.hours, labelled.phase, where="post", color=figures.INK, lw=1.2, ls=(0, (4, 3)), label="embryologist"
    )
    ax.set_yticks(range(len(PHASE_NAMES)), PHASE_NAMES)
    ax.set_xlabel("hours after fertilization")
    ax.set_xlim(0, g.hours.max())
    ax.set_title(f"Every development stage of one held-out embryo (video {video})")
    ax.legend(loc="lower right")
    figures.save(fig, "timelapse-stage-timeline")


def plot_errors(errors: pd.DataFrame, n_videos: int) -> None:
    order = [d for d in DIVISIONS if d in set(errors.phase)]
    data = [errors.loc[errors.phase == d, "error_h"].to_numpy() for d in order]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.axhspan(-1, 1, color=figures.COBALT, alpha=0.07, lw=0, label="within 1 hour")
    ax.axhline(0, color=figures.FAINT, lw=0.8)
    ax.boxplot(
        data,
        widths=0.55,
        whis=(10, 90),
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": to_rgba(figures.COBALT, 0.16), "edgecolor": figures.COBALT, "lw": 1.2},
        medianprops={"color": figures.COBALT, "lw": 2.2},
        whiskerprops={"color": figures.COBALT, "lw": 1.1},
        capprops={"color": figures.COBALT, "lw": 1.1},
    )
    names = [PHASE_NAMES[P[d]] for d in order]
    ax.set_xticks(range(1, len(order) + 1), names, rotation=30, ha="right")
    ax.set_ylabel("detected minus embryologist (hours)")
    ax.set_ylim(-Y_RANGE_H, Y_RANGE_H)
    ax.set_title(f"When each division happened: error on {n_videos} held-out videos")
    ax.legend(loc="upper left")
    ax.text(
        1.0,
        -0.36,
        f"Boxes: middle 50% of videos; whiskers: 10th-90th percentile, cut off at {Y_RANGE_H} hours.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
        color=figures.MUTED,
    )
    figures.save(fig, "timelapse-timing-errors")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=DEFAULT_RUN, help="stage-model run folder under runs/timelapse/")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    pred = pd.read_csv(paths.run_dir("timelapse", a.run) / f"predictions-{a.split}.csv")
    errors = timing_errors(pred)
    summary = errors.groupby("phase", sort=False).error_h.agg(
        videos="count", median_abs=lambda x: x.abs().median(), within_2h=lambda x: (x.abs() <= 2).mean()
    )
    print(summary.reindex([d for d in DIVISIONS if d in summary.index]).round(2).to_string())
    print(f"median absolute error, all divisions: {np.median(errors.error_h.abs()):.2f} h")
    plot_timeline(pred, typical_video(pred, errors))
    plot_errors(errors, pred.video.nunique())


if __name__ == "__main__":
    main()
