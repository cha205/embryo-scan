"""Train the development-stage model and measure its division-timing error on held-out videos.

Usage:
  python scripts/timelapse/train_stages.py                         # train + evaluate on val
  python scripts/timelapse/train_stages.py --eval-only --split test
"""

import argparse
import json

import pandas as pd
import torch

from embryoscan import paths
from embryoscan.common.reporting import md_table
from embryoscan.timelapse.labels import (
    DIVISIONS,
    decode_monotonic,
    meseguer_category,
    morphokinetics,
    onsets,
    timing_errors,
)
from embryoscan.timelapse.stages import DEFAULT_RUN, StageConfig, load_table, make_model, predict_frames, train


def evaluate(model, df: pd.DataFrame, run, split: str) -> dict:
    preds, cats = [], []
    for video, g in df.groupby("video", sort=False):
        logp = predict_frames(model, g["row"].to_numpy(), tta=True)
        dec = decode_monotonic(logp)
        h, truth = g["hours"].to_numpy(), g["phase"].to_numpy()
        preds.append(
            pd.DataFrame(
                {"video": video, "frame": g["frame"], "hours": h, "phase": truth, "raw": logp.argmax(1), "decoded": dec}
            )
        )
        on_t, on_p = onsets(truth, h), onsets(dec, h)
        cats.append((video, meseguer_category(morphokinetics(on_t)), meseguer_category(morphokinetics(on_p))))
    p = pd.concat(preds, ignore_index=True)
    p.to_csv(run / f"predictions-{split}.csv", index=False)
    m = p.phase >= 0
    e = timing_errors(p)
    e.to_csv(run / f"timing-errors-{split}.csv", index=False)
    c = pd.DataFrame(cats, columns=["video", "true", "pred"])
    per_phase = (
        e.groupby("phase", sort=False)["error_h"]
        .agg(
            videos="count",
            median_abs=lambda x: x.abs().median(),
            within_1h=lambda x: (x.abs() <= 1).mean(),
            within_2h=lambda x: (x.abs() <= 2).mean(),
        )
        .reindex([k for k in DIVISIONS if k in set(e.phase)])
    )
    both = c[(c.true != "X") & (c.pred != "X")]
    res = {
        "videos": int(df.video.nunique()),
        "frames": int(m.sum()),
        "frame_acc_raw": float((p.raw[m] == p.phase[m]).mean()),
        "frame_acc_decoded": float((p.decoded[m] == p.phase[m]).mean()),
        "frame_within_one_phase": float(((p.decoded[m] - p.phase[m]).abs() <= 1).mean()),
        "median_abs_error_h": float(e.error_h.abs().median()),
        "category_agreement": float((both.true == both.pred).mean()) if len(both) else float("nan"),
        "category_videos": int(len(both)),
    }
    (run / f"metrics-{split}.json").write_text(json.dumps(res, indent=2))
    text = [
        f"# Division timing: {split} videos\n",
        f"{res['videos']} videos from patients the model never saw, {res['frames']:,} labelled frames.\n",
        f"- Frame-by-frame phase accuracy: {res['frame_acc_raw']:.3f} raw, **{res['frame_acc_decoded']:.3f}** after "
        f"the never-go-backwards decoding ({res['frame_within_one_phase']:.3f} within one phase).",
        f"- Median timing error across all divisions: **{res['median_abs_error_h']:.2f} h** "
        "(frames are ~0.25 h apart).",
        f"- Meseguer 2011 category (A+ ... D-) from detected vs annotated timings: same category for "
        f"{res['category_agreement']:.0%} of {res['category_videos']} videos.\n",
        "Timing error per phase (detected onset minus the embryologist's):\n",
        md_table(per_phase.reset_index(), "{:.2f}"),
    ]
    paths.report("timelapse", f"divisions-{split}.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=DEFAULT_RUN, help="output folder under runs/timelapse/")
    ap.add_argument("--epochs", type=int, default=StageConfig.epochs)
    ap.add_argument("--arch", default=StageConfig.arch)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--eval-only", action="store_true", help="load the saved model instead of training")
    a = ap.parse_args()
    run = paths.run_dir("timelapse", a.run)
    run.mkdir(parents=True, exist_ok=True)
    df = load_table()
    log_file = open(run / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    if a.eval_only:
        model = make_model(a.arch, pretrained=False).cuda()
        model.load_state_dict(torch.load(run / "model.pt", map_location="cuda"))
    else:
        cfg = StageConfig(arch=a.arch, epochs=a.epochs)
        log(f"config: {cfg}")
        model = train(df[df.split == "train"], cfg, log=log, val=df[df.split == "val"])
        torch.save(model.state_dict(), run / "model.pt")
    model = model.to(memory_format=torch.channels_last)
    evaluate(model, df[df.split == a.split], run, a.split)


if __name__ == "__main__":
    main()
