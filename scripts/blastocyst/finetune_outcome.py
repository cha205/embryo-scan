"""Fine-tune the whole grading network on live birth (experiment).

Unlike run_outcome_models.py, which keeps the grading network frozen, this lets
it change what it looks at. The grade heads keep training on the embryologist's
grades as a side task. Uses the same patient-level 5-fold CV on the dev set.

Usage: python scripts/blastocyst/finetune_outcome.py [--epochs 12] [--init-fold 0]
"""

import argparse
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from embryoscan import paths
from embryoscan.blastocyst.grading import BACKGROUND, DEFAULT_RUN, Grader, ImageBank, class_weights, grade_loss
from embryoscan.blastocyst.outcome import TARGETS, load_outcome_table
from embryoscan.common.augment import dihedral_views, gpu_augment, to_net
from embryoscan.common.metrics import auc_diff_ci, binary_summary


class OutcomeNet(nn.Module):
    def __init__(self, grader: Grader):
        super().__init__()
        self.grader = grader
        self.outcome = nn.Linear(grader.backbone.num_features, 1)

    def forward(self, x):
        heads, f = self.grader(x, return_features=True)
        return heads, self.outcome(self.grader.drop(f)).squeeze(1)


def train_fold(init, X, y, g, epochs, batch, seed):
    torch.manual_seed(seed)
    dev = X.device
    net = OutcomeNet(init).to(dev)
    w_icm = class_weights(g[:, 1].cpu().numpy(), 3).to(dev)
    w_te = class_weights(g[:, 2].cpu().numpy(), 3).to(dev)
    opt = torch.optim.AdamW(
        [
            {"params": net.grader.parameters(), "lr": 2e-5},
            {"params": net.outcome.parameters(), "lr": 1e-3},
        ],
        weight_decay=0.05,
    )
    steps = len(X) // batch
    total = epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(1.0, i / total))))
    scaler = torch.amp.GradScaler()
    pos_weight = torch.tensor((1 - y.float().mean()) / y.float().mean(), device=dev)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X), device=dev)
        for i in range(steps):
            idx = perm[i * batch : (i + 1) * batch]
            x = to_net(gpu_augment(X[idx], BACKGROUND))
            with torch.autocast("cuda", dtype=torch.float16):
                heads, logit = net(x)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y[idx].float(), pos_weight=pos_weight)
            loss = loss + 0.5 * grade_loss(heads, g[idx], w_icm, w_te)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return net.eval()


@torch.no_grad()
def predict(net, X, batch=32):
    out = []
    for i in range(0, len(X), batch):
        x = to_net(X[i : i + batch].float() / 255)
        p = 0
        views = dihedral_views(x)
        for v in views:
            with torch.autocast("cuda", dtype=torch.float16):
                p = p + torch.sigmoid(net(v)[1].float())
        out.append((p / len(views)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=DEFAULT_RUN, help="grader run folder to start from (runs/blastocyst/)")
    ap.add_argument("--init-fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--target", default="lb", choices=list(TARGETS))
    a = ap.parse_args()

    df = load_outcome_table()
    dev = df[df.split == "dev"].reset_index(drop=True)
    bank = ImageBank()
    X_all = bank.get(dev.file.tolist(), "cuda")
    y_all = torch.as_tensor(dev[a.target].to_numpy(), device="cuda")
    g_np = np.stack([dev.exp_silver, dev.icm_silver, dev.te_silver], 1).astype(np.int64)
    g_all = torch.as_tensor(g_np, device="cuda")
    ckpt = paths.run_dir("blastocyst", a.run) / f"fold-{a.init_fold}.pt"
    label = TARGETS[a.target]

    oof = np.full(len(dev), np.nan)
    for k in sorted(dev.fold.unique()):
        tr = torch.as_tensor((dev.fold != k).to_numpy(), device="cuda")
        init = Grader(pretrained=False)
        init.load_state_dict(torch.load(ckpt, map_location="cuda"))
        net = train_fold(init.cuda(), X_all[tr], y_all[tr], g_all[tr], a.epochs, a.batch, seed=k)
        te = (dev.fold == k).to_numpy()
        oof[te] = predict(net, X_all[torch.as_tensor(te, device="cuda")])
        print(f"fold {k}: done", flush=True)
        del net, init
        torch.cuda.empty_cache()

    y = dev[a.target].to_numpy()
    groups = dev.patient.to_numpy()
    s = binary_summary(y, oof, groups)
    base = pd.read_csv(paths.run_dir("blastocyst", "baselines") / f"oof-{label}.csv")
    assert (base.file.to_numpy() == dev.file.to_numpy()).all()
    for name in ("age only", "embryologist's grade"):
        d, lo, hi, p = auc_diff_ci(y, oof, base[name].to_numpy(), groups)
        s[f"vs {name}"] = {"diff": d, "lo": lo, "hi": hi, "p_not_better": p}
    out = paths.run_dir("blastocyst", "outcome")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"file": dev.file, "fold": dev.fold, "y": y, "finetuned": oof}).to_csv(
        out / f"oof-{label}-finetuned.csv", index=False
    )
    (out / f"summary-{label}-finetuned.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
