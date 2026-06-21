"""Glottal-source features from the IAIF flow (DECISIONS #17-21):
GCT, CQ, NAQ, MFDR, SQ. Computed per voiced phone-segment: estimate the glottal
flow, detect glottal closure instants (GCIs) as minima of the flow derivative,
then per glottal cycle read the parameters and assign them to the frames the
cycle covers. Operational geometry (closed-phase threshold etc.) flagged for review.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from timit_features.config import Config, PHONE_CLASS, VOICED_CLASSES
from timit_features.io_timit import Segment
from timit_features import iaif

_VOICED = set(VOICED_CLASSES)
_CLOSED_FRAC = 0.10   # flow within 10% of its cycle floor counts as "closed"


def _ceiling(config: Config, sex: str) -> float:
    p = config.pitch
    if p.sex_dependent_f0 and sex.upper() == "M":
        return p.f0_ceiling_male_hz
    return p.f0_ceiling_female_hz


def _cycle_params(flow, d_dt, g0, g1, sr):
    """Return (GCT_ms, CQ, NAQ, MFDR, SQ) for one cycle [g0, g1)."""
    cyc = flow[g0:g1]
    if len(cyc) < 4:
        return [np.nan] * 5
    T0 = (g1 - g0) / sr
    f_ac = float(np.ptp(cyc))
    d_peak = float(np.max(-d_dt[g0:g1]))                  # MFDR magnitude (flow/s)
    naq = f_ac / (d_peak * T0) if d_peak > 0 and T0 > 0 else np.nan
    # closed phase: samples near the cycle floor
    thr = cyc.min() + _CLOSED_FRAC * (f_ac + 1e-12)
    closed = int(np.sum(cyc < thr))
    cq = closed / len(cyc)
    gct_ms = closed / sr * 1000.0
    # speed quotient: opening (min->max) vs closing (max->end)
    imin, imax = int(np.argmin(cyc)), int(np.argmax(cyc))
    open_t, close_t = (imax - imin), (len(cyc) - imax)
    sq = open_t / close_t if imax > imin and close_t > 0 else np.nan
    return [gct_ms, cq, naq, d_peak, sq]


def compute(signal: np.ndarray, frames, phones: list[Segment], sex: str,
            config: Config) -> dict[str, np.ndarray]:
    sr = config.framing.sample_rate_expected
    ceiling = _ceiling(config, sex)
    min_dist = max(2, int(sr / ceiling))
    n = frames.n_frames
    out = {k: np.full(n, np.nan) for k in ("GCT", "CQ", "NAQ", "MFDR", "SQ")}
    centers = frames.center_samples

    for s in phones:
        if PHONE_CLASS.get(s.label, "other") not in _VOICED:
            continue
        seg = np.asarray(signal[s.start:s.end], dtype=np.float64)
        if len(seg) < config.glottal.vt_lpc_order * 2:
            continue
        flow = iaif.glottal_flow(seg, sr, config.glottal)
        d_dt = np.gradient(flow) * sr
        gci, _ = find_peaks(-d_dt, distance=min_dist)
        for g0, g1 in zip(gci[:-1], gci[1:]):
            params = _cycle_params(flow, d_dt, g0, g1, sr)
            # global sample span of this cycle
            lo, hi = s.start + g0, s.start + g1
            in_cycle = (centers >= lo) & (centers < hi)
            if in_cycle.any():
                for key, val in zip(("GCT", "CQ", "NAQ", "MFDR", "SQ"), params):
                    out[key][in_cycle] = val
    return out
