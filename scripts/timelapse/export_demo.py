"""Export real held-out time-lapse videos for the web app: each video as one
WebM file plus the division times the model detected and the ones the
embryologist marked, and one held-out patient's embryos for the Compare tab.

Videos are picked by a fixed rule, not by how well the model did on them:
test-split videos that reach the expanded blastocyst stage and have ICM/TE
grades, in a seeded random order. The Compare patient is the first test
patient (by ID) with 3+ embryos recorded past day 5.

Frames are from Gomez et al. 2022 (CC BY-NC-SA 4.0); the exported folder
carries the same license.

Writes web/public/timelapse/{demo.json, LICENSE.txt, videos/<id>.webm,
compare/<id>-day3.jpg, compare/<id>-day5.jpg}.

Usage: python scripts/timelapse/export_demo.py [--run stages-resnet34] [--n 3]
"""

import argparse
import json
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from embryoscan import paths
from embryoscan.timelapse.labels import (
    PHASES,
    decode_monotonic,
    direct_cleavage,
    load_grades,
    meseguer_category,
    morphokinetics,
    onsets,
)
from embryoscan.timelapse.stages import DEFAULT_RUN, StageConfig, load_table, make_model, predict_frames

OUT = paths.WEB_PUBLIC / "timelapse"
FRAME_PX = 288
FPS = 10  # frame i of the WebM sits at i / FPS seconds; the app seeks to it


def frame_path(video: str, frame: int):
    hits = list((paths.GOMEZ_FRAMES / video).rglob(f"*RUN{frame}.jpeg"))
    return hits[0] if hits else None


def crop(video: str, frame: int, x0: int, y0: int) -> Image.Image | None:
    p = frame_path(video, frame)
    return None if p is None else Image.open(p).convert("L").crop((x0, y0, x0 + FRAME_PX, y0 + FRAME_PX))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=DEFAULT_RUN, help="stage-model run folder under runs/timelapse/")
    ap.add_argument("--arch", default=StageConfig.arch)
    ap.add_argument("--n", type=int, default=3, help="number of demo videos")
    a = ap.parse_args()
    run = paths.run_dir("timelapse", a.run)

    df = load_table()
    grades = load_grades().dropna()
    test = df[df.split == "test"]
    full = test.groupby("video").phase.max() >= PHASES.index("tEB")
    candidates = sorted(set(full[full].index) & set(grades.video))
    rng = np.random.default_rng(7)
    rng.shuffle(candidates)
    picked = candidates[: a.n]
    model = make_model(a.arch, pretrained=False).cuda()
    model.load_state_dict(torch.load(run / "model.pt", map_location="cuda"))
    model = model.to(memory_format=torch.channels_last).eval()

    shutil.rmtree(OUT, ignore_errors=True)  # rebuild from scratch: no stale files
    (OUT / "videos").mkdir(parents=True)
    (OUT / "compare").mkdir()
    (OUT / "LICENSE.txt").write_text(
        "Frames: Gomez et al. 2022, Human embryo time-lapse video dataset, "
        "https://doi.org/10.5281/zenodo.7912264, CC BY-NC-SA 4.0.\n"
    )
    videos = []
    for vid in picked:
        g = test[test.video == vid]
        logp = predict_frames(model, g["row"].to_numpy(), tta=True)
        dec = decode_monotonic(logp)
        h = g["hours"].to_numpy()
        on_ai, on_ex = onsets(dec, h), onsets(g["phase"].to_numpy(), h)
        m_ai, m_ex = morphokinetics(on_ai), morphokinetics(on_ex)
        x0, y0 = int(g.crop_x.iloc[0]), int(g.crop_y.iloc[0])
        writer = cv2.VideoWriter(
            str(OUT / "videos" / f"{vid}.webm"),
            cv2.VideoWriter_fourcc(*"VP90"),
            FPS,
            (FRAME_PX, FRAME_PX),
            isColor=False,
        )
        kept_hours, kept_phase = [], []
        for frame, hours, ph in zip(g.frame, h, dec, strict=True):
            im = crop(vid, int(frame), x0, y0)
            if im is None:
                continue
            writer.write(np.asarray(im))
            kept_hours.append(round(float(hours), 2))
            kept_phase.append(PHASES[int(ph)])
        writer.release()
        gr = grades[grades.video == vid].iloc[0]
        videos.append(
            {
                "id": vid,
                "video_url": f"/timelapse/videos/{vid}.webm",
                "fps": FPS,
                "hours": kept_hours,
                "ai_phase": kept_phase,
                "events_ai": {k: round(v, 2) for k, v in on_ai.items()},
                "events_expert": {k: round(v, 2) for k, v in on_ex.items()},
                "category_ai": meseguer_category(m_ai),
                "category_expert": meseguer_category(m_ex),
                "direct_cleavage_ai": direct_cleavage(m_ai),
                "grade": {"icm": gr.ICM, "te": gr.TE},
            }
        )
        print(
            f"{vid}: {len(kept_hours)} frames, "
            f"category AI {videos[-1]['category_ai']} / expert {videos[-1]['category_expert']}"
        )

    vi = pd.read_csv(run / "viability-scores-test.csv")
    per_patient = vi.groupby("patient").video.count()
    patients = per_patient[per_patient >= 3].sort_index().index
    compare = None
    if len(patients):
        pid = patients[0]
        embryos = []
        for _, r in vi[vi.patient == pid].sort_values("score", ascending=False).iterrows():
            g = test[test.video == r.video]
            x0, y0 = int(g.crop_x.iloc[0]), int(g.crop_y.iloc[0])
            thumbs = {}
            for day, hours in (("day3", 72.0), ("day5", 116.0)):
                k = int(np.argmin(np.abs(g.hours.to_numpy() - hours)))
                im = crop(r.video, int(g.frame.iloc[k]), x0, y0)
                if im is not None:
                    im.resize((160, 160)).save(OUT / "compare" / f"{r.video}-{day}.jpg", quality=85)
                    thumbs[day] = f"/timelapse/compare/{r.video}-{day}.jpg"
            embryos.append(
                {
                    "id": r.video,
                    "score": round(float(r.score), 3),
                    "category": r.category,
                    "events_ai": {
                        k: (None if pd.isna(r[k]) else round(float(r[k]), 2)) for k in ("t2", "t3", "t4", "t5", "t8")
                    },
                    "blastocyst": int(r.blastocyst),
                    "thumbs": thumbs,
                }
            )
        compare = {"patient": pid, "embryos": embryos}
        print(f"compare: patient {pid}, {len(embryos)} embryos")
    (OUT / "demo.json").write_text(json.dumps({"videos": videos, "compare": compare}, indent=1))


if __name__ == "__main__":
    main()
