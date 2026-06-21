"""Fano null: shuffled labels → acc ≈ 1/S, bounds ≈ 0 (small synthetic corpus)."""
import numpy as np

from src.fano import make_static_design, run_fold, cv_splits, fano_bounds


def _synthetic_df(S=16, n=10, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(S):
        center = rng.standard_normal(3)
        for _ in range(n):
            v = max(rng.random() - 0.3, 0.0)            # some exact zeros (state-2)
            rows.append(dict(speaker_id=f"S{i:03d}", sex="M" if i % 2 else "F",
                             A=center[0] + 0.2 * rng.standard_normal(),
                             B=center[1] + 0.2 * rng.standard_normal(),
                             VFI=v))
    return pd.DataFrame(rows)


def _cfg():
    return dict(speaker_key="speaker_id", sex_key="sex",
                feature_names=["A", "B", "VFI"], transforms={"VFI": "hurdle_lognonzero"},
                transform_default="linear", vfp={"name": "VFI"},
                fano=dict(cv_folds=5, cv_seed=0, classifiers=["logreg"]), seed=0)


def test_fano_null_collapses_to_chance():
    df, cfg = _synthetic_df(), _cfg()
    S = df.speaker_id.nunique()
    D, names, vfp_raw, sp = make_static_design(df, cfg)
    y = df.speaker_id.to_numpy()
    tr, te = cv_splits(y, 5, 0)[0]
    rng = np.random.default_rng(0)
    rn = run_fold(D, names, vfp_raw, sp, y, tr, te, S, "logreg", shuffle_y=rng.permutation(y))
    assert rn["acc"] < 5.0 / S                 # ≈ chance 1/S
    assert rn["I_fano"] < 0.5                  # bound ≈ 0


def test_fano_bounds_formula_perfect_classifier():
    # acc=1 ⇒ I_fano = H(Y) = log2 S exactly
    S = 8
    y = np.arange(S)
    proba = np.eye(S)
    out = fano_bounds(y, y, proba, np.arange(S), S)
    assert abs(out["I_fano"] - np.log2(S)) < 1e-9
    assert abs(out["I_xent"] - np.log2(S)) < 1e-9
