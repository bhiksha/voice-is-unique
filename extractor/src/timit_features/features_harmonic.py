"""Harmonic / voice-quality group: IHI, SHR, GNE, dCPP, AMD.

Faithfully implemented here: dCPP, IHI, SHR (Sun-style ratio), GNE (Michaelis-style
band-envelope correlation), AMD (<20 Hz envelope modulation).

VFI [62] and Nasality [48] are NOT produced here — each has a dedicated module
(deepfry_creak.py and features_nasality.py) and is filled in by extract.py.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from timit_features.config import Config, PHONE_CLASS, VOICED_CLASSES
from timit_features.io_timit import Segment
from timit_features import dsp

_VOICED = set(VOICED_CLASSES)
_EPS = 1e-12


def _ihi_shr(power, freqs, f0, max_hz=5000.0):
    """Per-frame IHI (mean |peak−k·F0| in Hz) and SHR (subharmonic/harmonic amp)."""
    n = power.shape[0]
    ihi = np.full(n, np.nan)
    shr = np.full(n, np.nan)
    mag = np.sqrt(power)
    df = freqs[1] - freqs[0]
    for i in range(n):
        f = f0[i]
        if not np.isfinite(f) or f <= 0:
            continue
        K = int(min(max_hz, freqs[-1]) // f)
        if K < 2:
            continue
        devs, harm, sub = [], 0.0, 0.0
        for k in range(1, K + 1):
            target = k * f
            lo = int((target * 0.9) / df); hi = int((target * 1.1) / df) + 1
            lo, hi = max(0, lo), min(len(freqs), hi)
            if hi <= lo:
                continue
            pk = lo + int(np.argmax(mag[i, lo:hi]))
            devs.append(abs(freqs[pk] - target))
            harm += mag[i, pk]
            sb = (k - 0.5) * f                       # subharmonic position
            j = int(round(sb / df))
            if 0 <= j < len(freqs):
                sub += mag[i, j]
        if devs:
            ihi[i] = float(np.mean(devs))
            shr[i] = sub / (harm + _EPS)
    return ihi, shr


def _gne_segment(seg, sr, band=(300.0, 4000.0), bw=1000.0, hop=300.0) -> float:
    """Michaelis-style GNE: max cross-correlation of Hilbert envelopes of
    LPC-residual band-pass signals across bands >= bw/2 apart."""
    import librosa
    if len(seg) < 64:
        return np.nan
    try:
        a = librosa.lpc(np.ascontiguousarray(seg), order=min(13, len(seg) // 2 - 1))
        res = filtfilt([1.0], [1.0], seg) if len(a) < 2 else np.convolve(seg, a, mode="same")
    except Exception:
        res = seg
    centers = np.arange(band[0] + bw / 2, band[1] - bw / 2 + 1, hop)
    envs = []
    for fc in centers:
        lo, hi = (fc - bw / 2) / (sr / 2), (fc + bw / 2) / (sr / 2)
        if lo <= 0 or hi >= 1:
            continue
        b, a2 = butter(4, [lo, hi], btype="band")
        env = np.abs(hilbert(filtfilt(b, a2, res)))
        envs.append((fc, env - env.mean()))
    best = np.nan
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            if abs(envs[i][0] - envs[j][0]) < bw / 2:
                continue
            a_, b_ = envs[i][1], envs[j][1]
            denom = np.sqrt((a_ @ a_) * (b_ @ b_)) + _EPS
            c = float((a_ @ b_) / denom)
            best = c if not np.isfinite(best) else max(best, c)
    return best


def _amd(rms: np.ndarray, speech_mask: np.ndarray, fps: float, max_hz=20.0) -> float:
    """Amplitude Modulation Depth: coefficient of variation of the <20 Hz
    envelope over speech frames (proxy for slow envelope fluctuation [36])."""
    env = rms[speech_mask]
    if env.size < 5 or env.mean() <= 0:
        return np.nan
    ny = fps / 2.0
    if max_hz < ny:
        b, a = butter(2, max_hz / ny, btype="low")
        env = filtfilt(b, a, env)
    return float(np.std(env) / (np.mean(np.abs(env)) + _EPS))


def compute(signal, frames, f0, cpp, phones: list[Segment], config: Config) -> dict:
    sr = config.framing.sample_rate_expected
    n = frames.n_frames
    power, freqs = dsp.power_spectrum(signal, config)
    if power.shape[0] != n:                       # guard alignment
        power = power[:n]

    ihi, shr = _ihi_shr(power, freqs, f0)

    # dCPP: frame-wise absolute change in CPP
    dcpp = np.full(n, np.nan)
    finite = np.isfinite(cpp)
    dcpp[1:] = np.where(finite[1:] & finite[:-1], np.abs(np.diff(cpp)), np.nan)

    # GNE per voiced segment → frames
    gne = np.full(n, np.nan)
    centers = frames.center_samples
    for s in phones:
        if PHONE_CLASS.get(s.label, "other") not in _VOICED:
            continue
        seg = np.asarray(signal[s.start:s.end], dtype=np.float64)
        in_seg = (centers >= s.start) & (centers < s.end)
        if in_seg.any():
            gne[in_seg] = _gne_segment(seg, sr)

    # AMD: single value broadcast to speech frames
    rms = dsp.frame_rms(signal, config)
    fps = sr / int(round(config.framing.hop_ms * 1e-3 * sr))
    amd_val = _amd(rms, frames.speech, fps)
    amd = np.where(frames.speech, amd_val, np.nan)

    return {
        "IHI": ihi, "SHR": shr, "GNE": gne, "dCPP": dcpp, "AMD": amd,
    }
