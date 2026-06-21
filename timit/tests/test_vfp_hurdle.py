"""Acceptance tests for the VFP hurdle and the Option C Σ_b reconstruction (§5).

A synthetic hurdle feature is generated with known presence probabilities, a known
magnitude distribution, and known correlations to other synthetic features. We
confirm:
  (1) the law-of-total-covariance decomposition reproduces the DIRECT between-speaker
      moments of the composite v_i = r_i·M_i (it is an identity → tight tolerance);
  (2) the small-sample Option C estimate of Var_b/Cov_b and the resulting PR recover
      the large-sample ("fully observed") truth within sampling error.
If this fails, no headline PR may be produced from the reconstruction.
"""
import numpy as np

from src.common import pr_from_cov
from src.vfp_hurdle import fit_hurdle, decompose_var, decompose_cov, vfp_sigma_b_row


def _generate(S, n, seed):
    """Hurdle-censored generative model. Returns (speaker_ids, vfp_values, Xbar)."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(S)                      # latent speaker factor
    pi = 1.0 / (1.0 + np.exp(-(1.2 * g - 0.4)))     # presence prob in (0,1)
    mu = 0.9 * g + 0.4 * rng.standard_normal(S)     # speaker log-magnitude mean
    loadings = np.array([1.0, 0.7, -0.5, 0.2])      # 4 other features ~ g
    p = len(loadings)

    spk_ids, vfp_vals = [], []
    Xsum = np.zeros((S, p))
    for i in range(S):
        present = rng.random(n) < pi[i]
        logv = mu[i] + 0.5 * rng.standard_normal(n)
        v = np.where(present, np.exp(logv), 0.0)    # state-1 exp, state-2 zero
        spk_ids.extend([i] * n)
        vfp_vals.extend(v.tolist())
        X = loadings[None, :] * g[i] + 0.6 * rng.standard_normal((n, p))
        Xsum[i] = X.mean(axis=0)
    return np.array(spk_ids), np.array(vfp_vals), Xsum


def _sigma_b_optionc(h, Xbar):
    """Assemble the (1+p)x(1+p) Σ_b with VFP as row 0 via Option C; X block direct."""
    var_v, cov_vX = vfp_sigma_b_row(h, Xbar)
    Xc = Xbar - Xbar.mean(axis=0, keepdims=True)
    SX = (Xc.T @ Xc) / Xbar.shape[0]
    p = Xbar.shape[1]
    Sig = np.zeros((p + 1, p + 1))
    Sig[0, 0] = var_v
    Sig[0, 1:] = cov_vX
    Sig[1:, 0] = cov_vX
    Sig[1:, 1:] = SX
    return Sig


def test_decomposition_matches_direct_moments():
    sp, vfp, Xbar = _generate(S=500, n=14, seed=1)
    h = fit_hurdle(sp, vfp)
    # Var: decomposition == population variance of v_i
    var_dec, _ = decompose_var(h)
    assert np.isclose(var_dec, h.v.var(), rtol=1e-10, atol=1e-12)
    # Cov: decomposition == direct cov(v, x_j) for every feature
    for j in range(Xbar.shape[1]):
        x = Xbar[:, j]
        direct = float(((h.v - h.v.mean()) * (x - x.mean())).mean())
        assert np.isclose(decompose_cov(h, x), direct, rtol=1e-10, atol=1e-12), j


def test_option_c_pr_recovers_truth():
    # small sample (estimation) vs large sample ("fully observed" truth)
    sp_s, vfp_s, Xb_s = _generate(S=500, n=14, seed=2)
    sp_l, vfp_l, Xb_l = _generate(S=8000, n=40, seed=3)
    pr_small = pr_from_cov(_sigma_b_optionc(fit_hurdle(sp_s, vfp_s), Xb_s))
    pr_truth = pr_from_cov(_sigma_b_optionc(fit_hurdle(sp_l, vfp_l), Xb_l))
    # PR is in [1, 5]; recover truth within sampling error
    assert 1.0 <= pr_small <= 5.0
    assert abs(pr_small - pr_truth) < 0.25, (pr_small, pr_truth)


def test_option_c_equals_fully_observed_pr_identity():
    # Σ_b assembled via Option C decomposition must equal Σ_b assembled from the
    # direct composite column v_i (the decomposition is a covariance identity).
    sp, vfp, Xbar = _generate(S=600, n=16, seed=4)
    h = fit_hurdle(sp, vfp)
    Sig_c = _sigma_b_optionc(h, Xbar)
    # direct: stack [v_i, Xbar] and take population covariance
    M = np.column_stack([h.v, Xbar])
    Mc = M - M.mean(axis=0, keepdims=True)
    Sig_d = (Mc.T @ Mc) / M.shape[0]
    assert np.allclose(Sig_c, Sig_d, rtol=1e-9, atol=1e-11)
    assert np.isclose(pr_from_cov(Sig_c), pr_from_cov(Sig_d), rtol=1e-9)
