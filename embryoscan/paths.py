"""Project paths. Override the data location with the EMBRYOSCAN_DATA env var."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("EMBRYOSCAN_DATA", ROOT / "data"))
RAW_ZIP = DATA / "raw" / "Blastocyst_Dataset.zip"
RAW = DATA / "raw" / "Blastocyst_Dataset"
IMAGES = RAW / "Images"
PROCESSED = DATA / "processed"
GOMEZ = DATA / "raw" / "gomez"  # Gomez et al. 2022 time-lapse videos

SPLITS = ROOT / "splits"  # small CSVs, committed so every result is reproducible
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"  # checkpoints and predictions, not committed
CACHE = ROOT / ".cache"
