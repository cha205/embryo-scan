"""Viability ranking from the first 3 days of division timings.

Outcome: did the embryo form a blastocyst (reach tB) by day 5 (116 h)? Only
videos recorded for at least 116 h count, so a video that simply stopped early
(e.g. a day-3 transfer) isn't mistaken for an embryo that stalled. Blastocyst
formation is the standard viability proxy in time-lapse research; embryos that
stall can't be transferred at day 5.

Features use only what's known by day 3 (72 h): t2 ... t8 and the derived cc2
(2nd cell-cycle length), s2 (2nd division synchrony) and direct cleavage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .labels import CATEGORY_ORDER, P, direct_cleavage, meseguer_category, morphokinetics, onsets

DAY3, DAY5 = 72.0, 116.0
EARLY = ["t2", "t3", "t4", "t5", "t6", "t7", "t8"]
FEATURES = EARLY + ["cc2", "s2", "direct", "reached_t8"]


def features_from_onsets(on: dict[str, float]) -> dict:
    """Day-3 timing features (plus the Meseguer category) from phase onsets."""
    on = {k: v for k, v in on.items() if v <= DAY3}  # only what is known by day 3
    m = morphokinetics(on)
    row = {k: on.get(k, np.nan) for k in EARLY}
    row.update(cc2=m["cc2"], s2=m["s2"], direct=float(direct_cleavage(m)), reached_t8=float("t8" in on))
    row["category"] = meseguer_category(m)
    return row


def outcome_table(frames: pd.DataFrame) -> pd.DataFrame:
    """One row per video recorded past day 5: annotated day-3 features and
    whether a blastocyst formed by day 5."""
    rows = []
    for video, g in frames.groupby("video", sort=False):
        h, ph = g["hours"].to_numpy(), g["phase"].to_numpy()
        if h.max() < DAY5:
            continue
        r = features_from_onsets(onsets(ph, h))
        r.update(
            video=video,
            patient=g.patient.iloc[0],
            split=g.split.iloc[0],
            blastocyst=int(((ph >= P["tB"]) & (h <= DAY5)).any()),
        )
        rows.append(r)
    return pd.DataFrame(rows)


def make_model() -> Pipeline:
    """Logistic regression on the timing features (median-imputed, scaled)."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)
    )


def category_score(categories: pd.Series) -> np.ndarray:
    """Meseguer category as a score, higher = better: A+ -> 8 ... D- -> 1, X -> 0."""
    return categories.map(lambda c: len(CATEGORY_ORDER) - 1 - CATEGORY_ORDER.index(c)).to_numpy(float)
