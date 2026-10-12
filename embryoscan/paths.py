"""File locations used across the project. Set EMBRYOSCAN_DATA to keep the data elsewhere."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Downloaded datasets
DATA = Path(os.environ.get("EMBRYOSCAN_DATA", ROOT / "data"))
KROMP = DATA / "raw" / "kromp2023"  # blastocyst photos, Kromp et al. 2023
KROMP_ZIP = KROMP / "Blastocyst_Dataset.zip"
KROMP_IMAGES = KROMP / "Images"
GOMEZ = DATA / "raw" / "gomez2022"  # time-lapse videos, Gomez et al. 2022
GOMEZ_FRAMES = GOMEZ / "embryo_dataset"

# Caches built by the prepare scripts
BLASTOCYST_CACHE = DATA / "processed" / "blastocyst"
BLASTOCYST_GRADES = BLASTOCYST_CACHE / "grades.csv"
BLASTOCYST_CLINICAL = BLASTOCYST_CACHE / "clinical.csv"
BLASTOCYST_IMAGES = BLASTOCYST_CACHE / "images_u8_288x384.npy"
BLASTOCYST_IMAGE_INDEX = BLASTOCYST_CACHE / "images_index.csv"
TIMELAPSE_CACHE = DATA / "processed" / "timelapse"
TIMELAPSE_FRAMES = TIMELAPSE_CACHE / "frames_u8_160.npy"
TIMELAPSE_TABLE = TIMELAPSE_CACHE / "frames.csv"

# Patient-level splits, committed so results are reproducible
SPLITS = ROOT / "splits"
SPLIT_GRADING = SPLITS / "blastocyst-grading.csv"
SPLIT_OUTCOME = SPLITS / "blastocyst-outcome.csv"
SPLIT_TIMELAPSE = SPLITS / "timelapse-videos.csv"

# Outputs
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RUNS = ROOT / "runs"  # checkpoints, predictions and logs
CACHE = ROOT / ".cache"  # pretrained backbone weights
WEB_PUBLIC = ROOT / "web" / "public"  # assets exported for the web app


def run_dir(pipeline: str, name: str) -> Path:
    """runs/<pipeline>/<name>, e.g. runs/blastocyst/grader-convnext-tiny."""
    return RUNS / pipeline / name


def report(pipeline: str, name: str) -> Path:
    """reports/<pipeline>/<name>, creating the folder."""
    path = REPORTS / pipeline / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
