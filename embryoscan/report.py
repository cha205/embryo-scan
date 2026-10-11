"""Tiny helpers for writing markdown reports."""

from __future__ import annotations

import pandas as pd


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if v != v else floatfmt.format(v)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def ci(v: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{v:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"
