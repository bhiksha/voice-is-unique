"""MI tests: null → I_corrected≈0; leading term log2 S; weighted==simple when equiprobable."""
import numpy as np

from src.mi import mi_feature, _cond_entropy, feature_bins


def test_null_feature_zero_corrected():
    rng = np.random.default_rng(0)
    S, n = 50, 12
    y = np.repeat(np.arange(S), n)
    x = rng.standard_normal(S * n)            # feature independent of speaker
    r = mi_feature(x, y, S, 5, n_perm=200, seed=0)
    assert r["I_corrected"] < 0.05            # debiased MI of an uninformative feature ≈ 0


def test_leading_term_is_log2_S():
    # single-bin (uninformative) feature ⇒ I_raw = log2(S) - H(Y|all) = log2 S - H(Y).
    rng = np.random.default_rng(1)
    S, n = 40, 10
    y = np.repeat(np.arange(S), n)            # balanced ⇒ H(Y)=log2 S
    x = np.ones(S * n)                         # degenerate → one bin
    r = mi_feature(x, y, S, 5, n_perm=10, seed=0)
    assert r["n_bins"] == 1
    assert abs(r["I_raw"]) < 1e-9             # log2(S) - log2(S) = 0


def test_weighted_equals_simple_when_equiprobable():
    # exactly equal-sized bins ⇒ probability-weighted H(Y|X) == simple average of H(Y|b).
    rng = np.random.default_rng(2)
    nbin, per_bin = 5, 120
    bk = np.repeat(np.arange(nbin), per_bin)
    yk = rng.integers(0, 30, nbin * per_bin)
    def H(yb):
        c = np.unique(yb, return_counts=True)[1] / yb.size
        return -(c * np.log2(c)).sum()
    simple = np.mean([H(yk[bk == b]) for b in range(nbin)])
    assert abs(_cond_entropy(bk, yk) - simple) < 1e-12
