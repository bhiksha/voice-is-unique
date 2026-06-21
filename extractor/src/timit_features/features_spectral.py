"""Spectral and energy features (the self-contained group: no Praat/IAIF/DL).

Each function returns a per-frame value array aligned with framing.build_frames;
NaN marks frames where the value is undefined. Domain masking + the min-valid
guard + the fixed statistic are applied later by aggregate.aggregate_frame_feature.

Implements: spectral_skewness, spectral_kurtosis, spectral_entropy,
spectral_rolloff, spectral_flux, alpha_ratio, LHR, SPI, RMS, SSPF.
(AMD is in the temporal group; see TODO.)
"""
from __future__ import annotations

import numpy as np

from timit_features.config import Config
from timit_features import dsp

_EPS = 1e-12


def _spectral_moments(power: np.ndarray, freqs: np.ndarray):
    """Per-frame standardized 3rd (skewness) and 4th (kurtosis) spectral moments,
    treating normalized power as a distribution over frequency."""
    tot = power.sum(axis=1) + _EPS
    p = power / tot[:, None]
    centroid = (p * freqs[None, :]).sum(axis=1)
    d = freqs[None, :] - centroid[:, None]
    var = (p * d ** 2).sum(axis=1)
    std = np.sqrt(var) + _EPS
    skew = (p * d ** 3).sum(axis=1) / std ** 3
    kurt = (p * d ** 4).sum(axis=1) / std ** 4
    return skew, kurt


def _entropy(power: np.ndarray) -> np.ndarray:
    tot = power.sum(axis=1) + _EPS
    p = power / tot[:, None]
    return -(p * np.log(p + _EPS)).sum(axis=1)


def _rolloff(power: np.ndarray, freqs: np.ndarray, pct: float) -> np.ndarray:
    csum = np.cumsum(power, axis=1)
    thresh = pct * csum[:, -1][:, None]
    idx = (csum >= thresh).argmax(axis=1)
    return freqs[idx]


def _flux(power: np.ndarray) -> np.ndarray:
    """Frame-to-frame Euclidean change of the magnitude spectrum; frame 0 = NaN."""
    mag = np.sqrt(power)
    flux = np.full(mag.shape[0], np.nan)
    if mag.shape[0] >= 2:
        flux[1:] = np.sqrt(((mag[1:] - mag[:-1]) ** 2).sum(axis=1))
    return flux


def _ratio(power, freqs, num_band, den_band):
    num = dsp.band_energy(power, freqs, *num_band)
    den = dsp.band_energy(power, freqs, *den_band)
    return num / (den + _EPS)


def _sspf(power: np.ndarray, freqs: np.ndarray, lo: float = 2000.0) -> np.ndarray:
    """Per-frame peak (max-power) frequency in the sibilant band (>= lo Hz)."""
    sel = freqs >= lo
    if not sel.any():
        return np.full(power.shape[0], np.nan)
    sub = power[:, sel]
    return freqs[sel][sub.argmax(axis=1)]


def compute(signal: np.ndarray, config: Config) -> dict[str, np.ndarray]:
    power, freqs = dsp.power_spectrum(signal, config)
    s = config.spectral
    skew, kurt = _spectral_moments(power, freqs)
    out = {
        "spectral_skewness": skew,
        "spectral_kurtosis": kurt,
        "spectral_entropy": _entropy(power),
        "spectral_rolloff": _rolloff(power, freqs, s.rolloff_percent),
        "spectral_flux": _flux(power),
        "alpha_ratio": _ratio(power, freqs, s.alpha_high_band, s.alpha_low_band),
        "LHR": _ratio(power, freqs, s.lhr_low_band, s.lhr_high_band),
        "RMS": dsp.frame_rms(signal, config),
        "SSPF": _sspf(power, freqs),
    }
    # SPI: H&H high/low across a single crossover (high_over_low).
    hi = dsp.band_energy(power, freqs, s.spi_crossover_hz, freqs[-1] + 1.0)
    lo = dsp.band_energy(power, freqs, 0.0, s.spi_crossover_hz)
    out["SPI"] = hi / (lo + _EPS) if s.spi_direction == "high_over_low" else lo / (hi + _EPS)
    return out
