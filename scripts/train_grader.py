"""Train the grading model with 5-fold cross-fitting and benchmark it on the
multi-expert gold test set.

Each fold model trains on 4/5 of the pool and predicts the remaining 1/5, so
every pool image gets an out-of-fold grade. Gold and held-out images are graded
by the average of all 5 models (an ensemble).

Usage:
  python scripts/train_grader.py --run convnext_t                    # 5 folds + combine
  python scripts/train_grader.py --run smoke --folds 0 --epochs 1    # quick check
  python scripts/train_grader.py --run strict --folds 0 --exclude-gold-patients
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from embryoscan import paths
from embryoscan.grading import ImageBank, TrainConfig, decode, predict, train
from embryoscan.metrics import paper_grading_metrics

GRADE_COLS = ["file", "exp", "icm", "te"]


def targets(df: pd.DataFrame) -> np.ndarray:
    return df[["exp", "icm", "te"]].fillna(-1).astype(int).to_numpy()


def load_table() -> pd.DataFrame:
    grades = pd.read_csv(paths.PROCESSED / "grades.csv")
    split = pd.read_csv(paths.SPLITS / "grade_split.csv")
    return split.merge(grades[GRADE_COLS], on="file")


def combine(out, df: pd.DataFrame, log) -> dict:
    preds, feats = [], []
    for k in range(5):
        preds.append(pd.read_csv(out / f"fold{k}_pred.csv"))
        feats.append(np.load(out / f"fold{k}_feat.npy"))
    p = pd.concat(preds, ignore_index=True)
    f = pd.DataFrame(np.concatenate(feats))
    f["file"] = p["file"].to_numpy()
    prob_cols = [c for c in p.columns if c.startswith("p_")]
    final = decode(p.groupby("file", sort=True)[prob_cols].mean().reset_index())
    final = final.merge(df[["file", "split", "fold"]], on="file")
    final.to_csv(out / "predictions.csv", index=False)
    np.save(out / "features.npy", f.groupby("file", sort=True).mean().loc[final["file"]].to_numpy(np.float32))
    gold = df[df.split == "gold_test"]
    m = paper_grading_metrics(final[GRADE_COLS], gold[GRADE_COLS])
    pool = final[final.split == "pool"]
    oof = paper_grading_metrics(pool[GRADE_COLS], df[df.split == "pool"][GRADE_COLS])
    res = {"gold_ensemble": m, "pool_out_of_fold_vs_silver": oof}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    log("ENSEMBLE gold: " + " ".join(f"{c} {m[c]['accuracy']:.3f}" for c in ("exp", "icm", "te")))
    log("out-of-fold vs silver: " + " ".join(f"{c} {oof[c]['accuracy']:.3f}" for c in ("exp", "icm", "te")))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--batch", type=int, default=TrainConfig.batch)
    ap.add_argument("--arch", default=TrainConfig.arch)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--exclude-gold-patients", action="store_true",
                    help="drop training images from patients who have a gold test image")
    ap.add_argument("--subsample", type=int, default=0,
                    help="randomly keep this many training images (size-matched control)")
    a = ap.parse_args()

    out = paths.RUNS / "grader" / a.run
    out.mkdir(parents=True, exist_ok=True)
    df = load_table()
    gold = df[df.split == "gold_test"]
    bank = ImageBank()
    log_file = open(out / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"config: {vars(a)}")
    for k in a.folds:
        tr = df[(df.split == "pool") & (df.fold != k)]
        if a.exclude_gold_patients:
            tr = tr[~tr.gold_patient]
        if a.subsample:
            tr = tr.sample(n=a.subsample, random_state=k)
        oof = df[(df.split == "pool") & (df.fold == k)]
        cfg = TrainConfig(arch=a.arch, epochs=a.epochs, batch=a.batch, lr=a.lr, seed=k)
        log(f"== fold {k}: train {len(tr)} images, out-of-fold {len(oof)}")
        model = train(tr.file.tolist(), targets(tr), cfg, bank, log=log, monitor=(oof.file.tolist(), targets(oof)))
        torch.save(model.state_dict(), out / f"fold{k}.pt")
        files = df[(df.split != "pool") | (df.fold == k)].file.tolist()
        pred, feats = predict(model, files, bank)
        pred.to_csv(out / f"fold{k}_pred.csv", index=False)
        np.save(out / f"fold{k}_feat.npy", feats.astype(np.float32))
        m = paper_grading_metrics(pred[GRADE_COLS], gold[GRADE_COLS])
        log(f"fold {k} gold: " + " ".join(f"{c} {m[c]['accuracy']:.3f}" for c in ("exp", "icm", "te")))
        del model
        torch.cuda.empty_cache()

    if all((out / f"fold{k}_pred.csv").exists() for k in range(5)):
        combine(out, df, log)


if __name__ == "__main__":
    main()
