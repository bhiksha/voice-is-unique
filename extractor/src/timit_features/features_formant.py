"""Formant hybrid (DECISIONS #D): F1-F4 from DeepFormants, F5 + B1-B5 from
order-20 Burg LPC at native 16 kHz, with Burg poles matched to the DeepFormants
F1-F4 to reject spurious peaks.

DeepFormants runs in the isolated `deepformants` conda env via subprocess (one
model-load per utterance). A temp standard-PCM WAV is written for it so it never
has to decode SPHERE.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from timit_features.config import Config, PHONE_CLASS, SONORANT_CLASSES
from timit_features import dsp

_SON = set(SONORANT_CLASSES)
_CONDA = os.path.expanduser("~/miniconda3/bin/conda")
_DF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                       "third_party", "deepformants")
_DF_INFER = os.path.join(_DF_DIR, "df_infer.py")


# ── Burg LPC (Collomb recursion) ────────────────────────────────────────────────

def burg(x: np.ndarray, order: int) -> np.ndarray:
    """LPC coefficients via Burg's method; a[0]=1, length order+1."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N <= order + 1:
        a = np.zeros(order + 1); a[0] = 1.0
        return a
    f = x.copy(); b = x.copy()
    a = np.zeros(order + 1); a[0] = 1.0
    Dk = 2.0 * np.dot(f, f) - f[0] ** 2 - b[-1] ** 2
    for k in range(order):
        if Dk <= 0:
            break
        num = np.dot(f[k + 1:N], b[0:N - k - 1])
        mu = -2.0 * num / Dk
        for n in range((k + 1) // 2 + 1):
            t1 = a[n] + mu * a[k + 1 - n]
            t2 = a[k + 1 - n] + mu * a[n]
            a[n], a[k + 1 - n] = t1, t2
        ff = f[k + 1:N].copy(); bb = b[0:N - k - 1].copy()
        f[k + 1:N] = ff + mu * bb
        b[0:N - k - 1] = bb + mu * ff
        Dk = (1.0 - mu * mu) * Dk - f[k + 1] ** 2 - b[N - k - 2] ** 2
    return a


def poles(a: np.ndarray, sr: int):
    """Return (freqs, bandwidths) for the resonances of LPC polynomial a, sorted
    by frequency, keeping 0 < f < sr/2 with positive bandwidth."""
    rts = np.roots(a)
    rts = rts[np.imag(rts) > 0]                       # one of each conj pair
    freqs = np.arctan2(np.imag(rts), np.real(rts)) * sr / (2 * np.pi)
    bws = -sr / np.pi * np.log(np.abs(rts) + 1e-12)
    keep = (freqs > 0) & (freqs < sr / 2) & (bws > 0)
    freqs, bws = freqs[keep], bws[keep]
    order_idx = np.argsort(freqs)
    return freqs[order_idx], bws[order_idx]


# ── DeepFormants (subprocess) ───────────────────────────────────────────────────

def _deepformants_segments(signal, sr, seg_windows):
    """Return (N,4) F1-F4 for each (begin,end) window via the deepformants env."""
    if not seg_windows:
        return np.empty((0, 4))
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "u.wav")
        sf.write(wav, np.asarray(signal, dtype=np.float32), sr, subtype="PCM_16")
        winf = os.path.join(d, "win.txt")
        with open(winf, "w") as fh:
            fh.write("\n".join(f"{b} {e}" for (b, e) in seg_windows))
        out = os.path.join(d, "out.csv")
        try:
            subprocess.run([_CONDA, "run", "-n", "deepformants", "python",
                            _DF_INFER, wav, "--windows", winf, "--out", out],
                           check=True, capture_output=True, timeout=600)
            rows = [list(map(float, ln.split(","))) for ln in open(out) if ln.strip()]
            return np.asarray(rows, dtype=np.float64)
        except Exception:
            return np.full((len(seg_windows), 4), np.nan)


# ── main ────────────────────────────────────────────────────────────────────────

def _ceiling(config, sex):
    f = config.formant
    return f.formant_ceiling_male_hz if sex.upper() == "M" else f.formant_ceiling_female_hz


def compute(signal, frames, phones, sex, config: Config) -> dict:
    sr = config.framing.sample_rate_expected
    n = frames.n_frames
    fc = config.formant
    ceiling = _ceiling(config, sex)
    names = ["F1", "F2", "F3", "F4", "F5", "B1", "B2", "B3", "B4", "B5"]
    out = {k: np.full(n, np.nan) for k in names}

    # sonorant segments → DeepFormants F1-F4 → assign to frames
    son_segs = [s for s in phones if PHONE_CLASS.get(s.label, "other") in _SON]
    windows = [(s.start / sr, s.end / sr) for s in son_segs]
    df = _deepformants_segments(signal, sr, windows)
    df_f = np.full((n, 4), np.nan)
    centers = frames.center_samples
    for si, s in enumerate(son_segs):
        if si >= len(df):
            break
        in_seg = (centers >= s.start) & (centers < s.end)
        df_f[in_seg] = df[si]

    # per sonorant frame: Burg poles, match to DF F1-F4 → B1-B4, then F5+B5
    xe = dsp.pre_emphasis(np.asarray(signal, dtype=np.float64), config.framing.pre_emphasis)
    fl = frames.frame_length
    for i in range(n):
        if not frames.sonorant[i] or not np.all(np.isfinite(df_f[i])):
            # still report DF formants if present
            for j in range(4):
                out[f"F{j+1}"][i] = df_f[i, j]
            continue
        st = frames.start_samples[i]
        frame = xe[st:st + fl]
        pf, pb = poles(burg(frame, fc.burg_order), sr)
        used = np.zeros(len(pf), dtype=bool)
        for j in range(4):
            out[f"F{j+1}"][i] = df_f[i, j]            # report DF freq
            if len(pf):
                k = int(np.argmin(np.abs(pf - df_f[i, j])))
                if not used[k] and abs(pf[k] - df_f[i, j]) <= fc.df_match_tolerance_hz:
                    used[k] = True
                    out[f"B{j+1}"][i] = pb[k]
        # F5: lowest unused pole above matched F4, below ceiling, narrow enough
        f4 = df_f[i, 3]
        cand = [(pf[k], pb[k]) for k in range(len(pf))
                if not used[k] and pf[k] > f4 and pf[k] < ceiling
                and pb[k] < fc.f5_max_bandwidth_hz]
        if cand:
            f5, b5 = min(cand, key=lambda t: t[0])
            out["F5"][i] = f5
            out["B5"][i] = b5
    return out
