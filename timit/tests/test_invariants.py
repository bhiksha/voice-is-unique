"""Invariants on the real table: no row deletion, NaN-ignored, transform-consistency, determinism."""
import numpy as np
import pytest

from src.common import load_config, load_table, transformed_matrix
from src import pr as PRm, mi as MIm, fano as FAm

CFG = "CONFIG/timit.json"


@pytest.fixture(scope="module")
def data():
    cfg = load_config(CFG)
    return cfg, load_table(cfg)


def test_transform_consistency(data):
    cfg, _ = data
    assert cfg["transforms"]["NAQ"] == "log"
    assert cfg["transforms"]["alpha_ratio"] == "log"
    assert cfg["transforms"]["LHR"] == "log"
    assert cfg["transforms"]["SPI"] == "log"
    assert cfg["transforms"]["VFI"].startswith("hurdle")
    for f in cfg["feature_names"]:
        if f not in ("NAQ", "alpha_ratio", "LHR", "SPI", "VFI"):
            assert cfg["transforms"].get(f, cfg["transform_default"]) == "linear"


def test_no_row_deletion(data):
    cfg, df = data
    # PR retains all speakers; Fano retains all rows; MI uses each feature's full support
    A = PRm.assemble(df, cfg)
    assert A["Xbar"].shape[0] == df[cfg["speaker_key"]].nunique() == 630
    D, names, vfp_raw, sp = FAm.make_static_design(df, cfg)
    assert D.shape[0] == len(df) == 6300
    y = df[cfg["speaker_key"]].to_numpy()
    for f in ("VOT", "SSPF", "F0"):
        r = MIm.mi_feature(df[f].to_numpy(float), y, 630, 5, n_perm=5, seed=0)
        assert r["n"] == int(np.isfinite(df[f].to_numpy(float)).sum())


def test_nan_ignored_no_missing_bin_no_indicator(data):
    cfg, df = data
    y = df[cfg["speaker_key"]].to_numpy()
    # MI: a feature's measured values fixed, adding more state-3 NaNs leaves I unchanged
    vot = df["VOT"].to_numpy(float)
    r1 = MIm.mi_feature(vot, y, 630, 5, n_perm=50, seed=0)
    vot2 = vot.copy()
    measured = np.isfinite(vot2)
    extra = np.where(measured)[0][:100]
    keep_vals = vot2.copy()
    vot2[extra] = np.nan                                   # blank 100 measured cells
    r2 = MIm.mi_feature(vot2, y, 630, 5, n_perm=50, seed=0)
    assert r2["n"] == r1["n"] - 100                         # excluded, not binned
    # Fano design has NO missing-indicator column: width == #features (incl VFP) ... + presence
    D, names, _, _ = FAm.make_static_design(df, cfg)
    assert D.shape[1] == len(cfg["feature_names"])          # 39 non-VFP + VFP_logmag = 40
    assert not any("indicator" in n or "missing" in n for n in names)


def test_determinism(data):
    cfg, df = data
    a = PRm.pr_point(df, cfg, vfp="C")["PR"]
    b = PRm.pr_point(df, cfg, vfp="C")["PR"]
    assert a == b
    y = df[cfg["speaker_key"]].to_numpy()
    m1 = MIm.mi_feature(df["F0"].to_numpy(float), y, 630, 5, n_perm=50, seed=0)
    m2 = MIm.mi_feature(df["F0"].to_numpy(float), y, 630, 5, n_perm=50, seed=0)
    assert m1["I_corrected"] == m2["I_corrected"]
