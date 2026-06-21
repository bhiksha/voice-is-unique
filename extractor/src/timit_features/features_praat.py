"""Praat (parselmouth) features: F0, semitone_SD_F0, jitter, shimmer, CPP.

F0 is a true per-frame contour (autocorrelation) sampled at frame centers.
jitter/shimmer/CPP need several glottal periods, so they are computed per voiced
phone-segment (Praat over the segment's time range) and assigned to that segment's
frames; the frame→utterance mean then weights by segment duration. Operational
choice — flagged for end-of-build review.
"""
from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call

from timit_features.config import Config, VOICED_CLASSES, PHONE_CLASS
from timit_features.io_timit import Segment

_VOICED = set(VOICED_CLASSES)
# Praat perturbation period bounds (s) and factors — standard Voice Report values.
_PMIN, _PMAX, _MAXFAC, _MAXAMP = 0.0001, 0.02, 1.3, 1.6


def _bounds(config: Config, sex: str) -> tuple[float, float]:
    p = config.pitch
    if p.sex_dependent_f0 and sex.upper() == "M":
        return p.f0_floor_male_hz, p.f0_ceiling_male_hz
    if p.sex_dependent_f0:
        return p.f0_floor_female_hz, p.f0_ceiling_female_hz
    return p.f0_floor_male_hz, p.f0_ceiling_female_hz


def _f0_contour(snd, centers_t, floor, ceil, config) -> np.ndarray:
    p = config.pitch
    pitch = call(snd, "To Pitch (ac)...", p.pitch_time_step_ms * 1e-3, floor,
                 p.max_candidates, 0, p.silence_threshold, p.voicing_threshold,
                 p.octave_cost, p.octave_jump_cost, p.voiced_unvoiced_cost, ceil)
    pr_t = np.asarray(pitch.xs())
    pr_f = np.asarray(pitch.selected_array["frequency"])  # 0 == unvoiced
    out = np.full(len(centers_t), np.nan)
    if len(pr_t):
        idx = np.clip(np.searchsorted(pr_t, centers_t), 0, len(pr_t) - 1)
        # nearest frame
        left = np.clip(idx - 1, 0, len(pr_t) - 1)
        pick_left = np.abs(pr_t[left] - centers_t) < np.abs(pr_t[idx] - centers_t)
        idx = np.where(pick_left, left, idx)
        f = pr_f[idx]
        out = np.where(f > 0, f, np.nan)
    return out


def _seg_perturbation(snd, seg_t0, seg_t1, floor, ceil):
    """(jitter_local, shimmer_local, cpp) for one segment; NaN on failure/too-short."""
    jit = shim = cpp = np.nan
    try:
        part = snd.extract_part(from_time=seg_t0, to_time=seg_t1, preserve_times=True)
        pp = call(part, "To PointProcess (periodic, cc)", floor, ceil)
        n = call(pp, "Get number of points")
        if n >= 5:
            jit = call(pp, "Get jitter (local)", 0, 0, _PMIN, _PMAX, _MAXFAC)
            shim = call([part, pp], "Get shimmer (local)", 0, 0, _PMIN, _PMAX, _MAXFAC, _MAXAMP)
    except Exception:
        pass
    try:
        pc = call(snd.extract_part(from_time=seg_t0, to_time=seg_t1, preserve_times=True),
                  "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
        cpp = call(pc, "Get CPPS", "no", 0.01, 0.001, 60.0, 330.0, 0.05,
                   "parabolic", 0.001, 0.05, "Straight", "Robust")
    except Exception:
        pass
    return (float(jit) if jit == jit else np.nan,
            float(shim) if shim == shim else np.nan,
            float(cpp) if cpp == cpp else np.nan)


def compute(signal: np.ndarray, frames, phones: list[Segment], sex: str,
            config: Config) -> dict[str, np.ndarray]:
    sr = config.framing.sample_rate_expected
    floor, ceil = _bounds(config, sex)
    snd = parselmouth.Sound(np.asarray(signal, dtype=np.float64), sampling_frequency=sr)
    centers_t = frames.center_samples / sr
    n = frames.n_frames

    f0 = _f0_contour(snd, centers_t, floor, ceil, config)
    jitter = np.full(n, np.nan)
    shimmer = np.full(n, np.nan)
    cpp = np.full(n, np.nan)

    for s in phones:
        if PHONE_CLASS.get(s.label, "other") not in _VOICED:
            continue
        t0, t1 = s.start / sr, s.end / sr
        in_seg = (centers_t >= t0) & (centers_t < t1)
        if not in_seg.any():
            continue
        j, sh, c = _seg_perturbation(snd, t0, t1, floor, ceil)
        jitter[in_seg] = j
        shimmer[in_seg] = sh
        cpp[in_seg] = c

    return {
        "F0": f0,
        "semitone_SD_F0": f0.copy(),   # aggregated with sd_semitone
        "jitter": jitter,
        "shimmer": shimmer,
        "CPP": cpp,
    }
