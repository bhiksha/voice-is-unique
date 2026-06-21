"""Corpus-agnostic core for the PR / summed-MI / Fano analysis.

Loads the utterance-level feature table, applies the per-feature transform dict
from CONFIG, and z-scores corpus-wide. Speaker grouping and the three data states
(1: measured>0, 2: measured==0 SIGNAL, 3: NaN missing) are exposed here; the VFP
hurdle (state-2 zeros carried separately) lives in ``vfp_hurdle.py``.

Nothing here listwise-deletes a row: NaNs (state 3) are preserved in the matrices
and excluded per-feature by the consumer (MI bins non-missing only; PR uses
pairwise; Fano mean-imputes with no indicator).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LOG_TRANSFORM = "log"


def load_config(path) -> dict:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def load_table(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(Path(cfg["input_parquet"]).expanduser())


def state_codes(values: np.ndarray) -> np.ndarray:
    """Per-cell state: 1 measured>0, 2 measured==0 (signal), 3 NaN (missing)."""
    s = np.full(values.shape, 3, dtype=np.int8)
    finite = np.isfinite(values)
    s[finite & (values > 0)] = 1
    s[finite & (values == 0)] = 2
    return s


def transform_of(name: str, cfg: dict) -> str:
    return cfg["transforms"].get(name, cfg["transform_default"])


def zscore(v: np.ndarray) -> np.ndarray:
    """Corpus-wide z-score over finite entries; NaNs preserved. ddof=0."""
    m = np.isfinite(v)
    mu = v[m].mean()
    sd = v[m].std()
    return (v - mu) / (sd if sd > 0 else 1.0)


def transformed_matrix(df: pd.DataFrame, cfg: dict):
    """Utterance-level transformed + z-scored matrix for every NON-VFP feature.

    Returns (X, names): X is (N, p) with NaNs preserved (state-3 kept missing, not
    imputed). The VFP hurdle feature is excluded here and handled in vfp_hurdle.
    Log features take natural log of their (positive) support before z-scoring.
    """
    vfp = cfg["vfp"]["name"]
    names = [f for f in cfg["feature_names"] if f != vfp]
    cols = []
    for f in names:
        v = df[f].to_numpy(dtype=float)
        if transform_of(f, cfg) == LOG_TRANSFORM:
            v = np.log(v)                     # state-3 NaNs stay NaN
        cols.append(zscore(v))
    return np.column_stack(cols), names


def speaker_means(values: np.ndarray, speaker_ids: np.ndarray, speakers: np.ndarray):
    """Per-speaker mean over that speaker's NON-missing utterances (NaN-aware).

    Returns (S,) with NaN where a speaker has no measured value for the feature
    (so PR can treat it pairwise instead of globally deleting the speaker).
    """
    out = np.full(len(speakers), np.nan)
    for k, s in enumerate(speakers):
        sub = values[speaker_ids == s]
        sub = sub[np.isfinite(sub)]
        if sub.size:
            out[k] = sub.mean()
    return out


def pr_from_cov(Sigma: np.ndarray) -> float:
    """Participation ratio PR = (Σλ)²/Σλ² of the correlation matrix of Sigma."""
    d = np.sqrt(np.clip(np.diag(Sigma), 1e-300, None))
    R = Sigma / np.outer(d, d)
    w = np.clip(np.linalg.eigvalsh(R), 0.0, None)
    return float(w.sum() ** 2 / np.square(w).sum())


def sorted_speakers(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    return np.array(sorted(df[cfg["speaker_key"]].unique()))


def fisher_components(values: np.ndarray, speaker_ids: np.ndarray,
                      speakers: np.ndarray) -> dict:
    """One-way random-effects variance components on a feature (NaN-aware).

    Returns Fstar = sigma_b2/sigma_w2, ICC, and the components, using only
    measured (finite) utterances; n_i recomputed on the measured support so a
    feature's Fisher uses its full support without globally deleting speakers.
    """
    finite = np.isfinite(values)
    y = values[finite]
    g = speaker_ids[finite]
    N = y.size
    if N == 0:
        return dict(Fstar=np.nan, ICC=np.nan, sigma_b2=np.nan, sigma_w2=np.nan, n=0, S=0)
    grand = y.mean()
    ni, ssw, ssb = [], 0.0, 0.0
    for s in speakers:
        sub = y[g == s]
        if sub.size == 0:
            continue
        m = sub.mean()
        ni.append(sub.size)
        ssw += ((sub - m) ** 2).sum()
        ssb += sub.size * (m - grand) ** 2
    ni = np.array(ni)
    S = ni.size
    if S < 2 or N - S < 1:
        return dict(Fstar=np.nan, ICC=np.nan, sigma_b2=np.nan, sigma_w2=np.nan, n=N, S=S)
    msb = ssb / (S - 1)
    msw = ssw / (N - S)
    n0 = (N - (ni ** 2).sum() / N) / (S - 1)
    sb2 = max((msb - msw) / n0, 0.0)
    sw2 = msw
    fstar = sb2 / sw2 if sw2 > 0 else np.nan
    icc = sb2 / (sb2 + sw2) if (sb2 + sw2) > 0 else np.nan
    return dict(Fstar=fstar, ICC=icc, sigma_b2=sb2, sigma_w2=sw2, n=N, S=S)
