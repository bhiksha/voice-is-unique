"""Nasality (#23) via A1-P0 (Chen 1997) — a single-channel acoustic nasalization
correlate (no Nasometer, no model).

Per sonorant frame:
  A1 = peak spectral amplitude (dB) in a window around F1,
  P0 = peak spectral amplitude (dB) in the low-frequency nasal-murmur band,
  Nasality = A1 - P0   (lower ⇒ more nasal; sign configurable).
F1 comes from the formant stage (per-frame). NaN where F1 or a peak is unavailable.

Reference: Chen, M. Y. (1997). Acoustic correlates of English and French nasalized
vowels. JASA 102(4), 2360-2370.
"""
from __future__ import annotations

import numpy as np

from timit_features.config import Config
from timit_features import dsp

_EPS = 1e-12


def _peak_db(power_row, freqs, lo, hi):
    sel = (freqs >= lo) & (freqs < hi)
    if not sel.any():
        return np.nan
    return 10.0 * np.log10(power_row[sel].max() + _EPS)


def compute(signal: np.ndarray, frames, f1: np.ndarray, config: Config) -> np.ndarray:
    """Return per-frame A1-P0 (dB); valid on sonorant frames with a known F1."""
    nc = config.nasality
    power, freqs = dsp.power_spectrum(signal, config)
    n = frames.n_frames
    if power.shape[0] != n:
        power = power[:n]
    out = np.full(n, np.nan)
    lo_n, hi_n = nc.nasal_band

    for i in range(n):
        if not frames.sonorant[i] or not np.isfinite(f1[i]) or f1[i] <= 0:
            continue
        p0 = _peak_db(power[i], freqs, lo_n, hi_n)
        a1 = _peak_db(power[i], freqs, f1[i] * (1 - nc.a1_rel_halfwidth),
                      f1[i] * (1 + nc.a1_rel_halfwidth))
        if not (np.isfinite(p0) and np.isfinite(a1)):
            continue
        out[i] = (a1 - p0) if nc.sign == "a1_minus_p0" else (p0 - a1)
    return out
