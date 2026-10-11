"""One look for every report figure, matching the web app's palette.

Colours carry meaning across all figures (Python and MATLAB):
cobalt is the AI model, ink is the embryologist, amber is the classical
image-processing baseline, grey is everything else.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no windows
import matplotlib.pyplot as plt  # noqa: E402

from .. import paths  # noqa: E402

__all__ = ["AMBER", "COBALT", "FAINT", "GRID", "INK", "MUTED", "plt", "save"]

INK = "#0d1424"
MUTED = "#5b6577"
FAINT = "#8a93a3"
GRID = "#e3e6eb"
COBALT = "#2f5bff"
AMBER = "#e8890c"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Manrope", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": FAINT,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "figure.dpi": 100,
    }
)


def save(fig: plt.Figure, name: str) -> Path:
    """Write reports/figures/<name>.png and close the figure."""
    path = paths.FIGURES / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(paths.ROOT)}")
    return path
