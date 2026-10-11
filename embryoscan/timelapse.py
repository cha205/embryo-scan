"""Time-lapse videos (Gomez et al. 2022): labels, splits, decoding and timing.

Each video follows one embryo for up to ~5 days, one frame every ~15 minutes.
An embryologist marked the frame where each development phase starts:

  tPB2 polar body out   tPNa pronuclei appear   tPNf pronuclei fade
  t2 ... t8  2 to 8 cells   t9+ 9 or more cells   tM morula
  tSB start of blastulation   tB blastocyst   tEB expanded   tHB hatching

The model predicts a phase for every frame; decode_monotonic() then finds the
most likely sequence that never goes backwards (an embryo can't un-divide),
which turns noisy per-frame guesses into clean division times.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import paths

PHASES = ["tPB2", "tPNa", "tPNf", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+", "tM", "tSB", "tB", "tEB", "tHB"]
P = {p: i for i, p in enumerate(PHASES)}


def patient_of(video: str) -> str:
    """'ADM715-1-2' -> 'ADM715' (letters + digits before the first dash)."""
    m = re.match(r"([A-Za-z]+\d+)", video)
    return m.group(1) if m else video


def _find(pattern: str):
    hits = sorted(paths.GOMEZ.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {pattern} under {paths.GOMEZ}; run scripts/download_timelapse.py")
    return hits


def load_frames_table() -> pd.DataFrame:
    """One row per frame: video, patient, frame (1-based), hours, phase (index
    into PHASES, or -1 for frames before the first annotation)."""
    times = {p.name.removesuffix("_timeElapsed.csv"): p for p in _find("*_timeElapsed.csv")}
    rows = []
    for f in _find("*_phases.csv"):
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
    which = {p: "test" for p in pats[:n_test]} | {p: "val" for p in pats[n_test:n_test + n_val]}
    v = pd.DataFrame({"video": sorted(set(videos))})
    v["patient"] = v.video.map(patient_of)
    v["split"] = v.patient.map(lambda p: which.get(p, "train"))
    return v


def decode_monotonic(logp: np.ndarray) -> np.ndarray:
    """Best phase sequence that never decreases.

    logp: (frames, phases) log-probabilities. Dynamic programming:
    best[t, c] = logp[t, c] + max over c' <= c of best[t-1, c'].
    Phases may be skipped (some embryos go straight from 2 to 4 cells).
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
    """Hours at which the embryo first reached each phase *or a later one*.

    Phases can be skipped when two divisions happen between frames (2 cells
    straight to 4): then t3 = t4, i.e. the time it first had at least 3 cells.
    Phases never reached are left out."""
    out = {}
    valid = phase >= 0
    for c, name in enumerate(PHASES):
        idx = np.flatnonzero(valid & (phase >= c))
        if len(idx):
            out[name] = float(hours[idx[0]])
    return out


def morphokinetics(on: dict[str, float]) -> dict[str, float]:
    """Derived timing variables used in the embryology literature."""
    m = {k: on.get(k, np.nan) for k in ("t2", "t3", "t4", "t5", "t8", "tSB", "tB")}
    m["cc2"] = m["t3"] - m["t2"]  # length of the second cell cycle
    m["s2"] = m["t4"] - m["t3"]  # synchrony of the second division
    return m


CATEGORY_ORDER = ["A+", "A-", "B+", "B-", "C+", "C-", "D+", "D-", "X"]


def meseguer_category(m: dict[str, float]) -> str:
    """Hierarchical model from Meseguer et al. 2011 (Hum Reprod 26:2658), where
    implantation rates fell from A to D:
      1. t5 (5-cell time) inside 48.8-56.6 h  -> A/B, outside -> C/D
      2. s2 (t4 - t3, how in-sync the 2nd division is) <= 0.76 h -> A or C, else B or D
      3. cc2 (t3 - t2, second cell cycle) <= 11.9 h -> '+', else '-'
    The paper's morphology-based exclusions (fragmentation etc.) need annotations
    this dataset doesn't have. An embryo that never reached 5 cells is outside
    the t5 window; one that never reached 4 cells can't be scored ('X').
    Times are hours after fertilization."""
    t5, s2, cc2 = (m.get(k, np.nan) for k in ("t5", "s2", "cc2"))
    if np.isnan(s2) or np.isnan(cc2):
        return "X"
    in_window, in_sync = (not np.isnan(t5)) and 48.8 <= t5 <= 56.6, s2 <= 0.76
    cat = ("A" if in_sync else "B") if in_window else ("C" if in_sync else "D")
    return cat + ("+" if cc2 <= 11.9 else "-")


def direct_cleavage(m: dict[str, float]) -> bool:
    """1 -> 3 cells in under 5 hours (cc2 < 5 h), linked to very low
    implantation (Rubio et al. 2012, Fertil Steril 98:1458)."""
    return bool(m.get("cc2", np.nan) < 5)
