"""Incremental participation ratio (PR) on the between-speaker correlation matrix.

R_b is the correlation matrix of the per-speaker MEAN vectors (transformed +
z-scored features), estimated pairwise-complete (primary). For the top-k features
(ranked by F*): PR(k) = (Σλ)²/Σλ² of the eigenvalues of R_b^(k). Cross-checks:
90%-variance-cutoff dimension and spectral-entropy dimension exp(−Σ p ln p).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from timit_features.config import FEATURE_NAMES


def speaker_means(zdf: pd.DataFrame, speaker_key: str, cols=FEATURE_NAMES) -> pd.DataFrame:
    """Per-speaker mean vector; entry NaN only if that speaker has 0 non-NaN utts."""
    return zdf.groupby(speaker_key)[list(cols)].mean()


def corr_from_means(M: pd.DataFrame, feats, method: str = "pairwise"):
    """Between-speaker correlation R_b over `feats`. Returns (R, psd_repaired)."""
    sub = M[list(feats)]
    if method == "listwise":
        sub = sub.dropna(axis=0, how="any")
    cov = sub.cov()                          # pandas: pairwise-complete by default
    d = np.sqrt(np.clip(np.diag(cov.to_numpy()), 1e-300, None))
    R = cov.to_numpy() / np.outer(d, d)
    R = (R + R.T) / 2.0
    repaired = False
    w = np.linalg.eigvalsh(R)
    if w.min() < -1e-10:                     # not PSD → nearest-PSD (clip eigenvalues)
        repaired = True
        vals, vecs = np.linalg.eigh(R)
        R = (vecs * np.clip(vals, 0, None)) @ vecs.T
        dd = np.sqrt(np.clip(np.diag(R), 1e-300, None))
        R = R / np.outer(dd, dd)
        R = (R + R.T) / 2.0
    return R, repaired


def participation_ratio(eigvals: np.ndarray) -> float:
    l = np.asarray(eigvals, dtype=float)
    s2 = np.sum(l ** 2)
    return float((l.sum() ** 2) / s2) if s2 > 0 else float("nan")


def variance_cutoff_dim(eigvals: np.ndarray, frac: float = 0.90) -> int:
    l = np.sort(np.clip(eigvals, 0, None))[::-1]
    if l.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(l) / l.sum(), frac) + 1)


def spectral_entropy_dim(eigvals: np.ndarray) -> float:
    l = np.clip(eigvals, 0, None)
    if l.sum() <= 0:
        return float("nan")
    p = l / l.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def pr_curve(M: pd.DataFrame, ranked_feats, method: str = "pairwise") -> pd.DataFrame:
    """PR(k), 90%-cutoff, spectral-entropy dim, ΔPR for k=1..len(ranked_feats)."""
    rows = []
    prev = None
    for k in range(1, len(ranked_feats) + 1):
        R, rep = corr_from_means(M, ranked_feats[:k], method)
        w = np.linalg.eigvalsh(R)
        pr = participation_ratio(w)
        rows.append(dict(k=k, feature_added=ranked_feats[k - 1], PR=pr,
                         dPR=(pr - prev if prev is not None else np.nan),
                         dim90=variance_cutoff_dim(w), Hdim=spectral_entropy_dim(w),
                         psd_repaired=rep))
        prev = pr
    return pd.DataFrame(rows)
