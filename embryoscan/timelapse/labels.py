"""Labels, splits and division timing for the Gomez et al. 2022 time-lapse videos.

Annotated phases:
  tPB2 polar body out   tPNa pronuclei appear   tPNf pronuclei fade
  t2 ... t8 2 to 8 cells   t9+ 9 or more cells   tM morula
  tSB blastulation starts   tB blastocyst   tEB expanded   tHB hatching
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import paths

PHASES = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+", "tM", "tSB", "tB", "tEB", "tHB"]
P = {p: i for i, p in enumerate(PHASES)}
PHASE_NAMES = ["second polar body", "pronuclear appearance", "pronuclear fading", "2-cell", "3-cell", "4-cell",
               "5-cell", "6-cell", "7-cell", "8-cell", "≥9-cell", "morula", "start of blastulation",
               "full blastocyst", "expanded blastocyst", "hatching blastocyst"]  # fmt: skip
DIVISIONS = ["t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+", "tM", "tSB", "tB", "tEB"]  # onsets scored for timing


def patient_of(video: str) -> str:
    """'ADM715-1-2' -> 'ADM715' (letters + digits before the first dash)."""
    m = re.match(r"([A-Za-z]+\d+)", video)
    return m.group(1) if m else video


def _find(folder: str, pattern: str):
    hits = sorted((paths.GOMEZ / folder).rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {pattern} under {paths.GOMEZ / folder}; run scripts/timelapse/download_data.py")
    return hits


def load_frames_table() -> pd.DataFrame:
    """One row per frame: video, patient, frame (1-based), hours, phase (index
    into PHASES, or -1 for frames before the first annotation)."""
    times = {
        p.name.removesuffix("_timeElapsed.csv"): p for p in _find("embryo_dataset_time_elapsed", "*_timeElapsed.csv")
    }
    rows = []
    for f in _find("embryo_dataset_annotations", "*_phases.csv"):
        video = f.name.removesuffix("_phases.csv")
        t = pd.read_csv(times[video])
        ann = pd.read_csv(f, header=None, names=["phase", "start", "end"])
        label = np.full(len(t), -1)
        frame = t["frame_index"].to_numpy()
        for ph, s, e in ann.itertuples(index=False):
            label[(frame >= s) & (frame <= e)] = P[ph]
        rows.append(pd.DataFrame({"video": video, "frame": frame, "hours": t["time"].to_numpy(), "phase": label}))
    df = pd.concat(rows, ignore_index=True)
    df["patient"] = df["video"].map(patient_of)
    return df


def load_grades() -> pd.DataFrame:
    """Blastocyst ICM/TE grades (A/B/C) for the videos that have them."""
    g = pd.read_csv(paths.GOMEZ / "embryo_dataset_grades.csv")
    return g.rename(columns={"video_name": "video"})


def make_video_split(videos: pd.Series, seed: int = 2026, test: float = 0.2, val: float = 0.1) -> pd.DataFrame:
    """Patient-level train/val/test split of videos."""
    pats = np.array(sorted(set(videos.map(patient_of))))
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n_test, n_val = int(len(pats) * test), int(len(pats) * val)
    which = {p: "test" for p in pats[:n_test]} | {p: "val" for p in pats[n_test : n_test + n_val]}
    v = pd.DataFrame({"video": sorted(set(videos))})
    v["patient"] = v.video.map(patient_of)
    v["split"] = v.patient.map(lambda p: which.get(p, "train"))
    return v


def decode_monotonic(logp: np.ndarray) -> np.ndarray:
    """Most likely phase sequence that never goes backwards, since an embryo can't un-divide.

    logp is (frames, phases) log-probabilities. Dynamic programming:
    best[t, c] = logp[t, c] + max over c' <= c of best[t-1, c'].
    Phases may be skipped, as when an embryo goes straight from 2 to 4 cells.
    """
    T, C = logp.shape
    best = np.empty((T, C))
    arg = np.empty((T, C), dtype=np.int64)
    best[0] = logp[0]
    for t in range(1, T):
        prev = best[t - 1]
        run_max = np.maximum.accumulate(prev)
        run_arg = np.zeros(C, dtype=np.int64)
        for c in range(1, C):
            run_arg[c] = c if prev[c] >= run_max[c - 1] else run_arg[c - 1]
        best[t] = logp[t] + run_max
        arg[t] = run_arg
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(best[-1].argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = arg[t, path[t]]
    return path


def onsets(phase: np.ndarray, hours: np.ndarray) -> dict[str, float]:
    """Hour each phase, or any later one, was first reached.

    When two divisions fall between frames (2 cells straight to 4), t3 equals t4:
    the first time the embryo had at least 3 cells. Phases never reached are left out.
    """
    out = {}
    valid = phase >= 0
    for c, name in enumerate(PHASES):
        idx = np.flatnonzero(valid & (phase >= c))
        if len(idx):
            out[name] = float(hours[idx[0]])
    return out


def timing_errors(pred: pd.DataFrame) -> pd.DataFrame:
    """Detected minus annotated onset in hours, for every video and division
    reached in both. pred has one row per frame with columns video, hours,
    phase (annotated) and decoded (predicted)."""
    rows = []
    for video, g in pred.groupby("video", sort=False):
        h = g["hours"].to_numpy()
        truth, found = onsets(g["phase"].to_numpy(), h), onsets(g["decoded"].to_numpy(), h)
        rows += [(video, ph, found[ph] - truth[ph]) for ph in DIVISIONS if ph in truth and ph in found]
    return pd.DataFrame(rows, columns=["video", "phase", "error_h"])


def morphokinetics(on: dict[str, float]) -> dict[str, float]:
    """Derived timing variables used in the embryology literature."""
    m = {k: on.get(k, np.nan) for k in ("t2", "t3", "t4", "t5", "t8", "tSB", "tB")}
    m["cc2"] = m["t3"] - m["t2"]  # length of the second cell cycle
    m["s2"] = m["t4"] - m["t3"]  # synchrony of the second division
    return m


CATEGORY_ORDER = ["A+", "A-", "B+", "B-", "C+", "C-", "D+", "D-", "X"]


def meseguer_category(m: dict[str, float]) -> str:
    """Hierarchical category from Meseguer et al. 2011 (Hum Reprod 26:2658), where
    implantation rates fall from A to D. Times are hours after fertilization.
      1. t5 within 48.8-56.6 h: A or B, otherwise C or D
      2. s2 = t4 - t3 at most 0.76 h: A or C, otherwise B or D
      3. cc2 = t3 - t2 at most 11.9 h: '+', otherwise '-'
    The paper's morphology exclusions need annotations this dataset lacks. An
    embryo that never reached 5 cells is outside the t5 window; one that never
    reached 4 cells can't be scored ('X')."""
    t5, s2, cc2 = (m.get(k, np.nan) for k in ("t5", "s2", "cc2"))
    if np.isnan(s2) or np.isnan(cc2):
        return "X"
    in_window, in_sync = (not np.isnan(t5)) and 48.8 <= t5 <= 56.6, s2 <= 0.76
    cat = ("A" if in_sync else "B") if in_window else ("C" if in_sync else "D")
    return cat + ("+" if cc2 <= 11.9 else "-")


def direct_cleavage(m: dict[str, float]) -> bool:
    """1 to 3 cells in under 5 hours (cc2 < 5 h), linked to very low implantation
    rates (Rubio et al. 2012, Fertil Steril 98:1458)."""
    return bool(m.get("cc2", np.nan) < 5)
