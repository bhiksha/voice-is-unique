"""Scaling-sweep invariants (§D), validated on synthetic data (no corpus needed):
nested subsets, fixed clips-per-speaker, fixed full-set standardization, and that
the sweep runs and reports the log2 N ceiling."""
import numpy as np
import pandas as pd

from src import scaling as SC


def _cfg():
    return dict(
        speaker_key="speaker_id", sex_key="sex",
        feature_names=["A", "B", "C", "VFI"],
        transforms={"VFI": "hurdle_lognonzero"}, transform_default="linear",
        vfp={"name": "VFI"},
        pr={"sigma_b_floor": 1e-3, "n_boot": 50, "boot_seed": 0},
        mi={"nbins": [5], "ref_nbin": 5, "n_perm": 20, "perm_seed": 0},
        fano={"cv_folds": 5, "cv_seed": 0, "classifiers": ["logreg"]},
        scaling={"nested_order_seed": 0, "fixed_clips_per_speaker": 8},
        seed=0)


def _synthetic(n_per_sex=40, m=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g, base in [("M", 0.0), ("F", 1.0)]:
        for i in range(n_per_sex):
            fac = rng.standard_normal(3) + base
            for c in range(m):
                v = max(rng.random() - 0.25, 0.0)
                rows.append(dict(speaker_id=f"{g}{i:04d}", sex=g, clip_id=f"{g}{i:04d}_{c:03d}",
                                 A=fac[0] + 0.3 * rng.standard_normal(),
                                 B=fac[1] + 0.3 * rng.standard_normal(),
                                 C=fac[2] + 0.3 * rng.standard_normal(), VFI=v))
    return pd.DataFrame(rows)


def test_nested_subsets():
    df, cfg = _synthetic(), _cfg()
    order = SC.nested_order(df, cfg)
    s_small = set(SC.subset(df, cfg, 10, order, 8)[cfg["speaker_key"]].unique())
    s_big = set(SC.subset(df, cfg, 20, order, 8)[cfg["speaker_key"]].unique())
    assert s_small < s_big                                    # strict subset
    assert len(s_small) == 20 and len(s_big) == 40            # balanced 2·n


def test_fixed_clips_per_speaker():
    df, cfg = _synthetic(m=12), _cfg()
    order = SC.nested_order(df, cfg)
    for n in (10, 25):
        sub = SC.subset(df, cfg, n, order, m_clips=8)
        counts = sub.groupby(cfg["speaker_key"]).size()
        assert (counts == 8).all()                           # exactly m at every size


def test_fullset_standardization_is_fixed():
    df, cfg = _synthetic(), _cfg()
    s_full = SC.fullset_stats(df, cfg)
    # stats computed on the full set are independent of any subset size
    s_again = SC.fullset_stats(df, cfg)
    assert s_full == s_again
    assert "__vfp_mag__" in s_full and "A" in s_full


def test_sweep_runs_and_reports_ceiling():
    df, cfg = _synthetic(n_per_sex=40, m=10), _cfg()
    out = SC.sweep(df, cfg, grid=[10, 20], m_clips=8, condition="pooled")
    assert list(out["S"]) == [20, 40]
    assert np.allclose(out["ceiling"], np.log2(out["S"]))
    assert (out["PR"] >= 1).all() and (out["PR"] <= 4).all()
    assert np.isfinite(out["fano"]).all()
