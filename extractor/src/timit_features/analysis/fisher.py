"""Per-feature transforms, corpus z-scoring, and variance-components Fisher ratios.

Definitions follow the prompt exactly:
  x̄_i = mean_j x_ij ; s²_i = var_j (ddof=1)
  σ_w² = Σ_i (n_i−1)s²_i / Σ_i (n_i−1)             (speakers with n_i<2 give 0 dof)
  MS_between = Σ_i n_i (x̄_i − x̄)² / (S−1)          (x̄ = utterance-weighted grand mean)
  n_0 = (N − Σ_i n_i²/N) / (S−1)
  σ_b² = max(0, (MS_between − σ_w²) / n_0)
  F* = σ_b²/σ_w² ; ICC = σ_b²/(σ_b²+σ_w²) ; F_anova = MS_between/σ_w², p
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from timit_features.config import FEATURE_NAMES
from timit_features.analysis.config import AnalysisConfig


def apply_transforms(df: pd.DataFrame, cfg: AnalysisConfig):
    """Return (transformed_df, info). VFI zeros→NaN then log; NAQ/alpha/LHR/SPI log."""
    out = df.copy()
    info: dict[str, dict] = {}
    for feat, t in cfg.transform_per_feature.items():
        if t == "none":
            continue
        v = out[feat].to_numpy(dtype=float)
        if t == "log_nonzero":
            n_zero = int((v == 0).sum())
            v = np.where(v == 0, np.nan, v)
            info[feat] = {"transform": t, "zeros_to_nan": n_zero,
                          "n_after": int(np.isfinite(v).sum())}
            v = np.log(v)
        elif t == "log":
            finite = v[np.isfinite(v)]
            n_nonpos = int((finite <= 0).sum())
            info[feat] = {"transform": t, "n_nonpositive": n_nonpos}
            v = np.where(v > 0, np.log(v), np.nan)   # non-positive -> NaN (reported)
        else:
            raise ValueError(f"unknown transform {t!r}")
        out[feat] = v
    return out, info


def zscore(df: pd.DataFrame, cols=FEATURE_NAMES) -> pd.DataFrame:
    """Corpus-wide z-score to unit total variance (per column, over non-NaN)."""
    out = df.copy()
    for c in cols:
        v = out[c].to_numpy(dtype=float)
        m = np.nanmean(v)
        sd = np.nanstd(v, ddof=1)
        out[c] = (v - m) / sd if sd > 0 else v - m
    return out


def fisher_one(values: np.ndarray, speaker_ids: np.ndarray) -> dict:
    """Variance-components Fisher stats for one feature."""
    v = np.asarray(values, dtype=float)
    spk = np.asarray(speaker_ids)
    ok = np.isfinite(v)
    v, spk = v[ok], spk[ok]
    N = v.size
    out = dict(n=N, S=0, n0=np.nan, sigma_w2=np.nan, sigma_b2=np.nan, Fstar=np.nan,
               sqrtFstar=np.nan, ICC=np.nan, F_anova=np.nan, p_anova=np.nan,
               MS_between=np.nan, var_of_means=np.nan)
    if N == 0:
        return out
    df_ = pd.DataFrame({"v": v, "spk": spk})
    g = df_.groupby("spk")["v"]
    n_i = g.size().to_numpy()
    xbar_i = g.mean().to_numpy()
    s2_i = g.var(ddof=1).to_numpy()              # NaN for n_i<2
    S = n_i.size
    grand = float(v.mean())                       # utterance-weighted grand mean
    # pooled within
    dof = n_i - 1
    num_w = np.nansum(np.where(dof > 0, dof * s2_i, 0.0))
    den_w = dof[dof > 0].sum()
    sigma_w2 = (num_w / den_w) if den_w > 0 else np.nan
    # between
    MS_between = float(np.sum(n_i * (xbar_i - grand) ** 2) / (S - 1)) if S > 1 else np.nan
    n0 = float((N - np.sum(n_i ** 2) / N) / (S - 1)) if S > 1 else np.nan
    sigma_b2 = max(0.0, (MS_between - sigma_w2) / n0) if np.isfinite(MS_between) and \
        np.isfinite(sigma_w2) and n0 and n0 > 0 else np.nan
    Fstar = (sigma_b2 / sigma_w2) if (sigma_w2 and sigma_w2 > 0) else np.nan
    ICC = (sigma_b2 / (sigma_b2 + sigma_w2)) if np.isfinite(sigma_b2) and \
        (sigma_b2 + sigma_w2) > 0 else np.nan
    F_anova = (MS_between / sigma_w2) if (sigma_w2 and sigma_w2 > 0) else np.nan
    p = float(f_dist.sf(F_anova, S - 1, N - S)) if (np.isfinite(F_anova) and N > S) else np.nan
    out.update(S=S, n0=n0, sigma_w2=sigma_w2, sigma_b2=sigma_b2, Fstar=Fstar,
               sqrtFstar=(np.sqrt(Fstar) if np.isfinite(Fstar) else np.nan), ICC=ICC,
               F_anova=F_anova, p_anova=p, MS_between=MS_between,
               var_of_means=float(np.var(xbar_i, ddof=1)) if S > 1 else np.nan)
    return out


def fisher_table(zdf: pd.DataFrame, speaker_key: str, cols=FEATURE_NAMES) -> pd.DataFrame:
    spk = zdf[speaker_key].to_numpy()
    rows = [dict(feature=c, **fisher_one(zdf[c].to_numpy(float), spk)) for c in cols]
    return pd.DataFrame(rows).sort_values("Fstar", ascending=False).reset_index(drop=True)
