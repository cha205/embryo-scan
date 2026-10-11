"""Export real held-out time-lapse videos for the web app: cropped frames plus
the division times the model detected and the ones the embryologist marked.

Videos are picked by a fixed rule, not by how well the model did on them:
test-split videos that reach the expanded blastocyst stage and have ICM/TE
grades, in a seeded random order.

Frames are from Gomez et al. 2022 (CC BY-NC-SA 4.0); the exported folder
carries the same license.

Usage: python scripts/export_timelapse_demo.py --run r34 [--n 3]
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

from embryoscan import paths
from embryoscan.stages import load_table, make_model, predict_frames
from embryoscan.timelapse import (PHASES, decode_monotonic, direct_cleavage, load_grades, meseguer_category,
                                  morphokinetics, onsets)

OUT = paths.ROOT / "web" / "public" / "demo"
FRAME_PX = 288
EVERY = 2


def frame_path(video: str, frame: int):
    hits = list((paths.GOMEZ / "embryo_dataset" / video).rglob(f"*RUN{frame}.jpeg"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r34")
    ap.add_argument("--arch", default="resnet34.a1_in1k")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    df = load_table()
    grades = load_grades().dropna()
    test = df[df.split == "test"]
    full = test.groupby("video").phase.max() >= PHASES.index("tEB")
    candidates = sorted(set(full[full].index) & set(grades.video))
    rng = np.random.default_rng(7)
    rng.shuffle(candidates)
    picked = candidates[: a.n]
    model = make_model(a.arch, pretrained=False).cuda()
    model.load_state_dict(torch.load(paths.RUNS / "timelapse" / a.run / "model.pt", map_location="cuda"))
    model = model.to(memory_format=torch.channels_last).eval()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "LICENSE.txt").write_text("Frames: Gomez et al. 2022, Human embryo time-lapse video dataset, "
                                     "https://doi.org/10.5281/zenodo.7912264, CC BY-NC-SA 4.0.\n")
    videos = []
    for vid in picked:
        g = test[test.video == vid]
        logp = predict_frames(model, g["row"].to_numpy(), tta=True)
        dec = decode_monotonic(logp)
        h = g["hours"].to_numpy()
        on_ai, on_ex = onsets(dec, h), onsets(g["phase"].to_numpy(), h)
        m_ai, m_ex = morphokinetics(on_ai), morphokinetics(on_ex)
        (OUT / vid).mkdir(exist_ok=True)
        x0, y0 = int(g.crop_x.iloc[0]), int(g.crop_y.iloc[0])
        kept_hours, kept_phase = [], []
        for i, (frame, hours, ph) in enumerate(zip(g.frame, h, dec)):
            if i % EVERY:
                continue
            p = frame_path(vid, int(frame))
            if p is None:
                continue
            im = Image.open(p).convert("L").crop((x0, y0, x0 + FRAME_PX, y0 + FRAME_PX))
            im.save(OUT / vid / f"f_{len(kept_hours):04d}.jpg", quality=82)
            kept_hours.append(round(float(hours), 2))
            kept_phase.append(PHASES[int(ph)])
        gr = grades[grades.video == vid].iloc[0]
        videos.append({
            "id": vid, "frame_url": f"/demo/{vid}/f_{{i}}.jpg", "hours": kept_hours, "ai_phase": kept_phase,
            "events_ai": {k: round(v, 2) for k, v in on_ai.items()},
            "events_expert": {k: round(v, 2) for k, v in on_ex.items()},
            "category_ai": meseguer_category(m_ai), "category_expert": meseguer_category(m_ex),
            "direct_cleavage_ai": direct_cleavage(m_ai), "grade": {"icm": gr.ICM, "te": gr.TE},
        })
        print(f"{vid}: {len(kept_hours)} frames, category AI {videos[-1]['category_ai']} / expert {videos[-1]['category_expert']}")
    # Compare view: every test video of one held-out patient with several
    # embryos recorded past day 5, ranked by the viability model's score.
    vi = pd.read_csv(paths.RUNS / "timelapse" / a.run / "viability_test_ai.csv")
    per_patient = vi.groupby("patient").video.count()
    patient = per_patient[per_patient >= 3].sort_index().index
    compare = None
    if len(patient):
        pid = patient[0]
        rows = vi[vi.patient == pid].sort_values("score", ascending=False)
        embryos = []
        for _, r in rows.iterrows():
            g = test[test.video == r.video]
            x0, y0 = int(g.crop_x.iloc[0]), int(g.crop_y.iloc[0])
            thumbs = {}
            for label, hours in (("day3", 72.0), ("day5", 116.0)):
                k = int(np.argmin(np.abs(g.hours.to_numpy() - hours)))
                p = frame_path(r.video, int(g.frame.iloc[k]))
                if p is not None:
                    Image.open(p).convert("L").crop((x0, y0, x0 + FRAME_PX, y0 + FRAME_PX)).resize((160, 160)).save(
                        OUT / f"cmp_{r.video}_{label}.jpg", quality=85)
                    thumbs[label] = f"/demo/cmp_{r.video}_{label}.jpg"
            embryos.append({"id": r.video, "score": round(float(r.score), 3), "category": r.category,
                            "events_ai": {k: (None if pd.isna(r[k]) else round(float(r[k]), 2)) for k in ("t2", "t3", "t4", "t5", "t8")},
                            "blastocyst": int(r.blastocyst), "thumbs": thumbs})
        compare = {"patient": pid, "embryos": embryos}
        print(f"compare: patient {pid}, {len(embryos)} embryos")
    (OUT / "timelapse.json").write_text(json.dumps({"videos": videos, "compare": compare}, indent=1))


if __name__ == "__main__":
    main()
