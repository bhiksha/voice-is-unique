"""Shared DSP primitives: pre-emphasis, framing, per-frame power spectrum, bands.

All framing uses the same grid as framing.build_frames (start = i*hop), so the
spectral frames line up 1:1 with the phone-class masks.
"""
from __future__ import annotations

import numpy as np
from scipy.signal.windows import get_window

from timit_features.config import Config


def pre_emphasis(x: np.ndarray, coef: float) -> np.ndarray:
    if coef <= 0:
        return x.astype(np.float64)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    y[1:] = x[1:] - coef * x[:-1]
    return y


def frame_matrix(x: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    """Return (n_frames, frame_length); frame i = x[i*hop : i*hop+frame_length]."""
    n = len(x)
    if n < frame_length:
        return np.empty((0, frame_length), dtype=np.float64)
    n_frames = 1 + (n - frame_length) // hop
    idx = np.arange(frame_length)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def power_spectrum(x: np.ndarray, config: Config):
    """Per-frame power spectrum aligned with the phone-class frames.

    Returns (power, freqs): power is (n_frames, n_bins) float64, freqs (n_bins,).
    """
    fr = config.framing
    sr = fr.sample_rate_expected
    frame_length = int(round(fr.frame_length_ms * 1e-3 * sr))
    hop = int(round(fr.hop_ms * 1e-3 * sr))

    xe = pre_emphasis(np.asarray(x, dtype=np.float64), fr.pre_emphasis)
    frames = frame_matrix(xe, frame_length, hop)
    if frames.shape[0] == 0:
        return np.empty((0, frame_length // 2 + 1)), np.fft.rfftfreq(frame_length, 1.0 / sr)
    win = get_window(fr.window, frame_length, fftbins=True)
    spec = np.fft.rfft(frames * win[None, :], axis=1)
    power = (spec.real ** 2 + spec.imag ** 2)
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / sr)
    return power, freqs


def band_energy(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Per-frame summed power in [lo, hi) Hz. power: (n_frames, n_bins)."""
    sel = (freqs >= lo) & (freqs < hi)
    if not sel.any():
        return np.zeros(power.shape[0])
    return power[:, sel].sum(axis=1)


def frame_rms(x: np.ndarray, config: Config) -> np.ndarray:
    """Per-frame RMS on the raw (non-pre-emphasised) signal, aligned to frames."""
    fr = config.framing
    sr = fr.sample_rate_expected
    frame_length = int(round(fr.frame_length_ms * 1e-3 * sr))
    hop = int(round(fr.hop_ms * 1e-3 * sr))
    frames = frame_matrix(np.asarray(x, dtype=np.float64), frame_length, hop)
    if frames.shape[0] == 0:
        return np.empty(0)
    return np.sqrt(np.mean(frames ** 2, axis=1))
