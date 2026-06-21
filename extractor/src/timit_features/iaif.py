"""IAIF glottal inverse filtering (Alku 1992) — native numpy/scipy/librosa.

Estimates the glottal volume-velocity waveform from the speech signal, with no
EGG. Used by features_glottal for GCT/CQ/NAQ/MFDR/SQ. Deterministic.

Reference: P. Alku, "Glottal wave analysis with pitch synchronous iterative
adaptive inverse filtering," Speech Communication 11(2-3):109-118, 1992.
"""
from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import lfilter, butter, filtfilt

from timit_features.config import GlottalSourceConfig


def _lpc(x: np.ndarray, order: int) -> np.ndarray:
    if len(x) <= order + 1:
        a = np.zeros(order + 1); a[0] = 1.0
        return a
    return librosa.lpc(np.ascontiguousarray(x), order=order)


def _highpass(x: np.ndarray, sr: int, fc: float) -> np.ndarray:
    if fc <= 0:
        return x
    b, a = butter(2, fc / (sr / 2.0), btype="high")
    return filtfilt(b, a, x)


def glottal_flow(x: np.ndarray, sr: int, cfg: GlottalSourceConfig) -> np.ndarray:
    """Return the IAIF glottal-flow estimate, same length as x."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < cfg.vt_lpc_order * 2:
        return np.zeros_like(x)
    x = _highpass(x, sr, cfg.highpass_hz)
    R = cfg.lip_radiation_coef

    # 1st pass: crude glottal (order 1) → VT → integrate to flow
    a_g1 = _lpc(x, 1)
    x_v = lfilter(a_g1, [1.0], x)
    a_v = _lpc(x_v, cfg.vt_lpc_order)
    flow = lfilter(a_v, [1.0], x)
    flow = lfilter([1.0], [1.0, -R], flow)            # cancel lip radiation

    # refine
    for _ in range(max(1, cfg.n_iterations) - 1):
        a_g = _lpc(flow, cfg.glottal_lpc_order)
        x_g = lfilter(a_g, [1.0], x)
        a_v = _lpc(x_g, cfg.vt_lpc_order)
        flow = lfilter(a_v, [1.0], x)
        flow = lfilter([1.0], [1.0, -R], flow)
    return flow
