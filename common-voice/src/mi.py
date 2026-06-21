"""Summed MI — per-feature mutual information with speaker, debiased (§3).

Each feature is quantile-binned into Nbin equiprobable bins over its NON-MISSING
(non-state-3) utterances only. Genuine NaNs (state 3) are IGNORED — not binned, not
counted; there is NO missing bin, so a feature's MI is invariant to how many
state-3 cells it has. VFP is augmented with a DEDICATED zero-bin (state-2, a real
no-creak measurement) plus quantile bins of log(VFP) over the non-zero (state-1)
values.

Because bins need not be equiprobable (VFP zero-bin; degenerate quantiles), the
conditional entropy is probability-weighted:
    H(Y|X) = Σ_b P(b) H(Y|bin=b),  P(b) = n_b / N_nonmissing
    I(Y;X) = log2(S) − H(Y|X)              (speakers equiprobable ⇒ H(Y)=log2 S)
Debias with a seeded permutation null (identical binning):
    I_corrected = max(0, I_raw − mean(I_null))
Summed marginal MI Σ_f I_corrected is an UPPER bound on joint speaker information.
"""
from __future__ import annotations

import numpy as np

from src.common import state_codes


def _quantile_bins(x: np.ndarray, nbin: int) -> np.ndarray:
    """Equiprobable quantile bins; collapses gracefully on ties/degeneracy."""
    if x.size == 0:
        return np.zeros(0, dtype=int)             # no measured support (e.g. VOT under MFA)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbin + 1)))
    if edges.size < 3:
        return np.zeros(x.size, dtype=int)        # degenerate → single bin
    return np.digitize(x, edges[1:-1], right=False)


def feature_bins(vals: np.ndarray, nbin: int, is_vfp: bool = False):
    """Return (bins, keep). keep = non-missing (state-3 excluded). For VFP, bin 0 is
    the exact-zero (state-2) bin; bins 1.. are log-magnitude quantiles of state-1."""
    st = state_codes(vals)
    keep = st != 3
    b = np.full(vals.shape, -1, dtype=int)
    if is_vfp:
        b[st == 2] = 0
        s1 = st == 1
        b[s1] = _quantile_bins(np.log(vals[s1]), nbin) + 1
    else:
        b[keep] = _quantile_bins(vals[keep], nbin)
    return b, keep


def _cond_entropy(bins: np.ndarray, y: np.ndarray) -> float:
    N = bins.size
    H = 0.0
    for b in np.unique(bins):
        yb = y[bins == b]
        _, c = np.unique(yb, return_counts=True)
        p = c / c.sum()
        H += (yb.size / N) * (-(p * np.log2(p)).sum())
    return H


def mi_feature(vals, y, S, nbin, is_vfp=False, n_perm=200, seed=0) -> dict:
    bins, keep = feature_bins(vals, nbin, is_vfp)
    bk, yk = bins[keep], y[keep]
    if bk.size == 0:                              # no measured support → no information
        return dict(I_raw=0.0, I_null=0.0, I_corrected=0.0, n=0, n_bins=0,
                    n_missing=int(vals.size))
    HY = np.log2(S)
    I_raw = HY - _cond_entropy(bk, yk)
    rng = np.random.default_rng(seed)
    nulls = np.array([HY - _cond_entropy(bk, rng.permutation(yk)) for _ in range(n_perm)])
    I_null = float(nulls.mean())
    return dict(I_raw=float(I_raw), I_null=I_null,
                I_corrected=max(0.0, float(I_raw) - I_null),
                n=int(keep.sum()), n_bins=int(np.unique(bk).size),
                n_missing=int((~keep).sum()))


def mi_presence(vals, y, S, n_perm=200, seed=0) -> dict:
    """VFP zero-bin (presence/absence) contribution: binary state-2 vs state-1 MI."""
    st = state_codes(vals)
    keep = st != 3
    b = np.where(st[keep] == 2, 0, 1)
    yk = y[keep]
    HY = np.log2(S)
    I_raw = HY - _cond_entropy(b, yk)
    rng = np.random.default_rng(seed)
    nulls = np.array([HY - _cond_entropy(b, rng.permutation(yk)) for _ in range(n_perm)])
    return dict(I_raw=float(I_raw), I_null=float(nulls.mean()),
                I_corrected=max(0.0, float(I_raw) - float(nulls.mean())))
