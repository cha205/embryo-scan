"""Train the development-stage model on time-lapse videos, then detect division
times on held-out videos and measure how far off they are (in hours).

Usage:
  python scripts/train_timelapse.py --run r34                  # train + evaluate on val
  python scripts/train_timelapse.py --run r34 --eval-only --split test
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from embryoscan import paths
from embryoscan.report import md_table
from embryoscan.stages import StageConfig, load_table, make_model, predict_frames, train
from embryoscan.timelapse import PHASES, decode_monotonic, meseguer_category, morphokinetics, onsets

KEY = ["t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9+", "tM", "tSB", "tB", "tEB"]


def evaluate(model, df: pd.DataFrame, out_dir, split: str) -> dict:
    preds, errs, cats = [], [], []
    for video, g in df.groupby("video", sort=False):
        logp = predict_frames(model, g["row"].to_numpy(), tta=True)
        dec = decode_monotonic(logp)
        h, truth = g["hours"].to_numpy(), g["phase"].to_numpy()
        preds.append(pd.DataFrame({"video": video, "frame": g["frame"], "hours": h, "phase": truth,
                                   "raw": logp.argmax(1), "decoded": dec}))
        on_t, on_p = onsets(truth, h), onsets(dec, h)
        for ph in KEY:
            if ph in on_t and ph in on_p:
                errs.append((video, ph, on_p[ph] - on_t[ph]))
        cats.append((video, meseguer_category(morphokinetics(on_t)), meseguer_category(morphokinetics(on_p))))
    p = pd.concat(preds, ignore_index=True)
    p.to_csv(out_dir / f"pred_{split}.csv", index=False)
    m = p.phase >= 0
    e = pd.DataFrame(errs, columns=["video", "phase", "error_h"])
    c = pd.DataFrame(cats, columns=["video", "true", "pred"])
    per_phase = e.groupby("phase", sort=False)["error_h"].agg(
        videos="count", median_abs=lambda x: x.abs().median(), within_1h=lambda x: (x.abs() <= 1).mean(),
        within_2h=lambda x: (x.abs() <= 2).mean()).reindex([k for k in KEY if k in set(e.phase)])
    both = c[(c.true != "X") & (c.pred != "X")]
    res = {
        "videos": int(df.video.nunique()), "frames": int(m.sum()),
        "frame_acc_raw": float((p.raw[m] == p.phase[m]).mean()),
        "frame_acc_decoded": float((p.decoded[m] == p.phase[m]).mean()),
        "frame_within_one_phase": float(((p.decoded[m] - p.phase[m]).abs() <= 1).mean()),
        "median_abs_error_h": float(e.error_h.abs().median()),
        "category_agreement": float((both.true == both.pred).mean()) if len(both) else float("nan"),
        "category_videos": int(len(both)),
    }
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(res, indent=2))
    text = [f"# Division timing: {split} videos\n",
            f"{res['videos']} videos from patients the model never saw, {res['frames']:,} labelled frames.\n",
            f"- Frame-by-frame phase accuracy: {res['frame_acc_raw']:.3f} raw, **{res['frame_acc_decoded']:.3f}** after "
            f"the never-go-backwards decoding ({res['frame_within_one_phase']:.3f} within one phase).",
            f"- Median timing error across all divisions: **{res['median_abs_error_h']:.2f} h** (frames are ~0.25 h apart).",
            f"- Meseguer 2011 category (A+ ... D-) from detected vs annotated timings: same category for "
            f"{res['category_agreement']:.0%} of {res['category_videos']} videos.\n",
            "Timing error per phase (detected onset minus the embryologist's):\n",
            md_table(per_phase.reset_index(), "{:.2f}")]
    (paths.REPORTS / f"timelapse_{split}.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r34")
    ap.add_argument("--epochs", type=int, default=StageConfig.epochs)
    ap.add_argument("--arch", default=StageConfig.arch)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    out = paths.RUNS / "timelapse" / a.run
    out.mkdir(parents=True, exist_ok=True)
    df = load_table()
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    if a.eval_only:
        model = make_model(a.arch, pretrained=False).cuda()
        model.load_state_dict(torch.load(out / "model.pt", map_location="cuda"))
    else:
        cfg = StageConfig(arch=a.arch, epochs=a.epochs)
        log(f"config: {cfg}")
        model = train(df[df.split == "train"], cfg, log=log, val=df[df.split == "val"])
        torch.save(model.state_dict(), out / "model.pt")
    model = model.to(memory_format=torch.channels_last)
    evaluate(model, df[df.split == a.split], out, a.split)


if __name__ == "__main__":
    main()
