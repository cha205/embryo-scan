import numpy as np
import pandas as pd
import pytest

from embryoscan import paths
from embryoscan.common.metrics import auc_ci, auc_diff_ci, paper_grading_metrics


def test_auc_ci_perfect_and_random():
    y = np.array([0, 0, 1, 1] * 25)
    auc, lo, hi = auc_ci(y, y.astype(float), n_boot=200)
    assert auc == lo == hi == 1.0
    rng = np.random.default_rng(0)
    auc, lo, hi = auc_ci(y, rng.random(len(y)), n_boot=500)
    assert lo < 0.5 < hi


def test_cluster_bootstrap_is_wider_for_duplicated_patients():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = y * 0.3 + rng.random(200)
    yy, pp = np.repeat(y, 2), np.repeat(p, 2)  # every "patient" twice
    _, lo_i, hi_i = auc_ci(yy, pp, None, n_boot=500)
    _, lo_c, hi_c = auc_ci(yy, pp, np.repeat(np.arange(200), 2), n_boot=500)
    assert (hi_c - lo_c) > (hi_i - lo_i)


def test_auc_diff_identical_models_is_zero():
    y = np.array([0, 1] * 50)
    p = np.linspace(0, 1, 100)
    d, lo, hi, _ = auc_diff_ci(y, p, p, n_boot=100)
    assert d == lo == hi == 0


def test_paper_metrics_skip_icm_te_for_early_embryos():
    gold = pd.DataFrame({"file": ["a", "b", "c"], "exp": [3, 1, 2], "icm": [0, 3, 1], "te": [0, 3, 2]})
    pred = pd.DataFrame({"file": ["a", "b", "c"], "exp": [3, 1, 1], "icm": [0, 3, 3], "te": [1, 3, 3]})
    m = paper_grading_metrics(pred, gold)
    assert m["exp"]["accuracy"] == pytest.approx(2 / 3)
    assert m["icm"]["n"] == 1 and m["icm"]["accuracy"] == 1.0  # b and c skipped (expansion grade 1-2)
    assert m["te"]["accuracy"] == 0.0


@pytest.mark.skipif(not paths.SPLIT_OUTCOME.exists(), reason="run scripts/blastocyst/prepare_data.py first")
def test_blastocyst_splits_never_share_a_patient():
    o = pd.read_csv(paths.SPLIT_OUTCOME)
    assert not set(o[o.split == "dev"].patient) & set(o[o.split == "test"].patient)
    dev = o[o.split == "dev"]
    assert (dev.groupby("patient").fold.nunique() == 1).all()
    g = pd.read_csv(paths.SPLIT_GRADING)
    pool = g[g.split == "pool"]
    assert (pool.groupby("patient").fold.nunique() == 1).all()
    assert not set(pool.patient) & set(o[o.split == "test"].patient)


@pytest.mark.skipif(not paths.SPLIT_TIMELAPSE.exists(), reason="run scripts/timelapse/prepare_data.py first")
def test_timelapse_split_never_shares_a_patient():
    v = pd.read_csv(paths.SPLIT_TIMELAPSE)
    assert (v.groupby("patient").split.nunique() == 1).all()
    assert set(v.split) == {"train", "val", "test"}
