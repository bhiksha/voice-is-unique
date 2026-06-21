"""Frame→utterance aggregation with the MIN_VALID_FRAMES guard.

A frame-level feature contributes a per-frame value array (NaN where undefined).
Aggregation restricts to the feature's domain mask AND finite values, applies the
min-valid-frame guard, then reduces with the feature's fixed statistic.
"""
from __future__ import annotations

import numpy as np

from timit_features.config import Config


def aggregate_frame_feature(
    values: np.ndarray,
    domain_mask: np.ndarray,
    statistic: str,
    config: Config,
) -> tuple[float, int]:
    """Return (utterance_value, n_valid_frames).

    statistic ∈ {"median", "mean", "sd_semitone", "flux_mean"}.
    Below MIN_VALID_FRAMES valid frames → NaN (but n_valid is still reported).
    """
    values = np.asarray(values, dtype=np.float64)
    valid = domain_mask & np.isfinite(values)
    n_valid = int(valid.sum())
    if n_valid < config.aggregation.min_valid_frames:
        return float("nan"), n_valid

    v = values[valid]
    if statistic == "median":
        return float(np.median(v)), n_valid
    if statistic in ("mean", "flux_mean"):
        return float(np.mean(v)), n_valid
    if statistic == "sd_semitone":
        # SD of F0 expressed in semitones; ref cancels in an SD, use ddof=1.
        semis = 12.0 * np.log2(v / np.median(v))
        return (float(np.std(semis, ddof=1)) if n_valid >= 2 else float("nan")), n_valid
    raise ValueError(f"unknown aggregation statistic: {statistic!r}")
