"""Load and clean the Kromp et al. 2023 human blastocyst dataset.

The raw files need four fixes before they can be trusted:

1. Image names are zero-padded inconsistently ("187_1.png" in a CSV, "187_01.png"
   on disk, "0001_01.png" for early patients). Everything is matched on the
   (patient, embryo) integer pair instead of the string.
2. AMH and endometrium values were auto-converted to dates by a German-locale
   spreadsheet: "1.4" became "01.Apr" and "1.26" became "Jän.26".
   decode_amh() and decode_endo() reverse this (see their docstrings).
3. Some egg counts are written as sums ("10+1").
4. One silver-label image (838_02.png) is listed twice with conflicting grades.

Grade codes follow the paper: expansion 0-4 means Gardner grade 1-5; inner cell
mass (ICM) and trophectoderm (TE) 0/1/2 mean A/B/C, and 3 means "not defined"
(ICM and TE are only graded once the embryo reaches expansion grade 3).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import paths

MONTHS = {
    "jän": 1,
    "jan": 1,
    "feb": 2,
    "mär": 3,
    "mrz": 3,
    "apr": 4,
    "mai": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "okt": 10,
    "nov": 11,
    "dez": 12,
}
ND = 3  # ICM/TE "not defined" code
GRADE_LETTERS = "ABC"


def image_key(name: str) -> tuple[int, int]:
    """'187_1.png' -> (187, 1)."""
    m = re.fullmatch(r"\s*(\d+)_(\d+)\.png\s*", str(name))
    if not m:
        raise ValueError(f"unexpected image name: {name!r}")
    return int(m.group(1)), int(m.group(2))


def _date_parts(raw: str):
    """Return ('DD.Mon', day, month) or ('Mon.YY', month, yy) or None."""
    s = raw.strip()
    m = re.fullmatch(r"(\d{1,2})\.([^\W\d_]{3})", s)
    if m and m.group(2).lower() in MONTHS:
        return "DD.Mon", int(m.group(1)), MONTHS[m.group(2).lower()]
    m = re.fullmatch(r"([^\W\d_]{3})\.(\d{2})", s)
    if m and m.group(1).lower() in MONTHS:
        return "Mon.YY", MONTHS[m.group(1).lower()], int(m.group(2))
    return None


@dataclass(frozen=True)
class Decoded:
    value: float  # best estimate (NaN if missing)
    lo: float  # smallest plausible original value
    hi: float  # largest plausible original value
    how: str  # 'missing' | 'number' | 'DD.Mon' | 'Mon.YY'

    @property
    def ambiguous(self) -> bool:
        return not (math.isnan(self.lo) or self.lo == self.hi)


def decode_amh(raw: str) -> Decoded:
    """Undo spreadsheet date conversion of AMH values (ng/ml, 2 decimals).

    The source had 2-decimal values with trailing zeros dropped, so:
      "1.26" -> month 26 is invalid -> read as Jan 1926 -> "Jän.26"  => 1 + 26/100
      "1.4"  -> 1 April                                -> "01.Apr"  => 1.4
      "1.04" -> also 1 April                           -> "01.Apr"  => 1.04
      "2.11" -> 2 November                             -> "02.Nov"  => 2.11
    "DD.Mon" with month 1-9 is therefore ambiguous between D.M and D.0M; the
    estimate is the midpoint and lo/hi keep both readings for sensitivity checks.
    Evidence for the trailing-zero rule: no "Okt" (month 10) dates and no
    "Mon.YY" with YY in 20, 30, ... 90 occur anywhere in the file.
    """
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return Decoded(math.nan, math.nan, math.nan, "missing")
    if re.fullmatch(r"\d+(\.\d+)?", s):
        v = float(s)
        return Decoded(v, v, v, "number")
    parts = _date_parts(s)
    if parts is None:
        raise ValueError(f"cannot decode AMH value {raw!r}")
    kind, a, b = parts
    if kind == "Mon.YY":
        v = a + b / 100
        return Decoded(v, v, v, kind)
    day, month = a, b
    if month >= 10:
        v = day + month / 100
        return Decoded(v, v, v, kind)
    lo, hi = day + month / 100, day + month / 10
    return Decoded((lo + hi) / 2, lo, hi, kind)


def decode_endo(raw: str) -> Decoded:
    """Undo date conversion of endometrium thickness (mm, measured to 0.1 mm).

    "8.5" -> 8 May -> "08.Mai" => 8.5. Ultrasound thickness is recorded to one
    decimal, so "DD.Mon" decodes to D + M/10 without ambiguity.
    """
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return Decoded(math.nan, math.nan, math.nan, "missing")
    if re.fullmatch(r"\d+(\.\d+)?", s):
        v = float(s)
        return Decoded(v, v, v, "number")
    parts = _date_parts(s)
    if parts is None or parts[0] != "DD.Mon":
        raise ValueError(f"cannot decode endometrium value {raw!r}")
    v = parts[1] + parts[2] / 10
    return Decoded(v, v, v, parts[0])


def decode_count(raw: str) -> float:
    """'10+1' -> 11.0, '' -> NaN."""
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return math.nan
    if not re.fullmatch(r"\d+(\+\d+)*", s):
        raise ValueError(f"cannot decode count {raw!r}")
    return float(sum(int(x) for x in s.split("+")))


def _read_csv(name: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(paths.KROMP / name, sep=";", dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"unknown encoding for {name}")
    return df.loc[:, [c for c in df.columns if c.strip() and not c.startswith("Unnamed")]]


def image_index() -> pd.DataFrame:
    """One row per image file on disk: file, patient, embryo."""
    rows = [(p.name, *image_key(p.name)) for p in sorted(paths.KROMP_IMAGES.glob("*.png"))]
    if not rows:
        raise FileNotFoundError(f"no images in {paths.KROMP_IMAGES}; run scripts/blastocyst/download_data.py")
    return pd.DataFrame(rows, columns=["file", "patient", "embryo"])


def _attach_files(df: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    lookup = {(p, e): f for f, p, e in index[["file", "patient", "embryo"]].itertuples(index=False)}
    keys = df["Image"].map(image_key)
    df = df.assign(file=keys.map(lookup), patient=keys.map(lambda k: k[0]), embryo=keys.map(lambda k: k[1]))
    missing = df["file"].isna()
    if missing.any():
        raise ValueError(f"{missing.sum()} annotation rows have no image: {df.loc[missing, 'Image'].tolist()[:5]}")
    return df


def _gold_code(v: str) -> float:
    s = str(v).strip().upper()
    if s in ("", "NA", "NAN"):
        return np.nan  # not assessable
    if s == "ND":
        return float(ND)
    return float(int(float(s)))


def _agreement(v) -> float:
    s = str(v).strip()
    if re.fullmatch(r"\d+([.,]\d+)?", s):
        return float(s.replace(",", "."))
    return np.nan  # 'revised_cons' (re-annotated by consensus) or 'ND'


def load_grades() -> pd.DataFrame:
    """All 2,344 images with their label set and Gardner grade codes.

    label_set: 'silver' (one expert, training), 'gold' (multi-expert consensus,
    test), 'conflict' (838_02.png, two different silver labels), 'none' (846_01.png).
    """
    index = image_index()
    silver = _attach_files(_read_csv("Gardner_train_silver.csv"), index)
    dup = silver["file"].duplicated(keep=False)
    conflict_files = set(silver.loc[dup, "file"])
    silver = silver.loc[~dup]

    gold_x = pd.read_excel(paths.KROMP / "Gardner_test_gold.xlsx", dtype=str)
    gold_x = gold_x.loc[:, [c for c in gold_x.columns if not c.startswith("Unnamed")]]
    gold = _attach_files(gold_x, index)

    out = index.copy()
    out["label_set"] = "none"
    out.loc[out.file.isin(silver.file), "label_set"] = "silver"
    out.loc[out.file.isin(gold.file), "label_set"] = "gold"
    out.loc[out.file.isin(conflict_files), "label_set"] = "conflict"

    s = silver.set_index("file")
    g = gold.set_index("file")
    for crit in ("EXP", "ICM", "TE"):
        col = crit.lower()
        out[col] = np.nan
        m = out.file.isin(s.index)
        out.loc[m, col] = out.loc[m, "file"].map(s[f"{crit}_silver"]).astype(float)
        m = out.file.isin(g.index)
        out.loc[m, col] = out.loc[m, "file"].map(g[f"{crit}_gold"].map(_gold_code))
        out[f"{col}_agreement"] = (
            out.file.map(g[f"{crit}_Agreement"].map(_agreement)) if f"{crit}_Agreement" in g else np.nan
        )
    return out


def load_clinical() -> pd.DataFrame:
    """The 754 transferred embryos with cleaned clinical fields and outcomes.

    Columns: file, patient, embryo, day, age, amh, amh_lo, amh_hi, amh_how,
    amh_ambiguous, endo, endo_how, coc, mii, exp_silver, icm_silver, te_silver,
    ss (positive hCG), ha (ongoing pregnancy with heartbeat), lb (live birth),
    n_transferred (embryos from this patient in the file: 2 = double transfer).
    The undocumented 'Fond' column is dropped.
    """
    raw = _attach_files(_read_csv("Clincial_annotations.csv"), image_index())  # sic: file name typo
    amh = raw["AMH"].map(decode_amh)
    endo = raw["Endo"].map(decode_endo)
    df = pd.DataFrame(
        {
            "file": raw["file"],
            "patient": raw["patient"],
            "embryo": raw["embryo"],
            "day": raw["d"].astype(int),
            "age": raw["Age"].astype(float),
            "amh": [d.value for d in amh],
            "amh_lo": [d.lo for d in amh],
            "amh_hi": [d.hi for d in amh],
            "amh_how": [d.how for d in amh],
            "amh_ambiguous": [d.ambiguous for d in amh],
            "endo": [d.value for d in endo],
            "endo_how": [d.how for d in endo],
            "coc": raw["COC"].map(decode_count),
            "mii": raw["MII"].map(decode_count),
            "count_was_sum": raw["COC"].str.contains("+", regex=False) | raw["MII"].str.contains("+", regex=False),
            "exp_silver": raw["EXP_silver"].astype(int),
            "icm_silver": raw["ICM_silver"].astype(int),
            "te_silver": raw["TE_silver"].astype(int),
            "ss": raw["SS"].astype(int),
            "ha": raw["HA"].astype(int),
            "lb": raw["LB"].astype(int),
        }
    )
    df["n_transferred"] = df.groupby("patient")["file"].transform("count")
    return df.sort_values("file").reset_index(drop=True)


def gardner_label(exp: float, icm: float, te: float) -> str:
    """(3, 0, 1) -> '4AB'; ICM/TE 'not defined' -> just the expansion grade."""
    if np.isnan(exp):
        return "n/a"
    g = str(int(exp) + 1)
    if np.isnan(icm) or np.isnan(te) or int(icm) == ND or int(te) == ND:
        return g
    return g + GRADE_LETTERS[int(icm)] + GRADE_LETTERS[int(te)]
