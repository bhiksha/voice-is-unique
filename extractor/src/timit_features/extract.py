"""Per-utterance orchestration: run all feature groups, aggregate to the fixed
40-vector with a per-feature valid-frame coverage record and the CONFIG hash.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from timit_features.config import Config, CONFIG, FEATURE_ORDER, FEATURE_NAMES
from timit_features.io_timit import load_utterance, Utterance
from timit_features.framing import build_frames
from timit_features.aggregate import aggregate_frame_feature
from timit_features import (features_spectral, features_praat, features_glottal,
                            features_harmonic, features_formant, features_alignment,
                            features_nasality, deepfry_creak)


@dataclass
class UtteranceRecord:
    basename: str
    rel_path: str
    speaker_id: str
    sex: str
    dialect_region: str
    split: str
    sample_rate: int
    duration: float
    config_hash: str
    decode_ok: bool
    features: dict = field(default_factory=dict)   # name -> value (NaN allowed)
    coverage: dict = field(default_factory=dict)   # name -> n_valid_frames / n_events
    error: str | None = None

    def vector(self) -> np.ndarray:
        return np.array([self.features[n] for n in FEATURE_NAMES], dtype=np.float64)


def _all_nan_record(u: Utterance, config: Config) -> UtteranceRecord:
    return UtteranceRecord(
        basename=u.basename, rel_path=u.rel_path, speaker_id=u.speaker_id, sex=u.sex,
        dialect_region=u.dialect_region, split=u.split, sample_rate=u.sample_rate,
        duration=u.duration, config_hash=config.config_hash(), decode_ok=False,
        features={n: float("nan") for n in FEATURE_NAMES},
        coverage={n: 0 for n in FEATURE_NAMES}, error=u.error,
    )


def safe_extract_utterance(wav_path, timit_root, config: Config = CONFIG) -> UtteranceRecord:
    """extract_utterance that NEVER raises — any error yields an all-NaN record
    with the error recorded, so one bad utterance can't abort a corpus run."""
    try:
        return extract_utterance(wav_path, timit_root, config)
    except Exception as exc:
        try:
            r = _all_nan_record(load_utterance(wav_path, timit_root), config)
            r.error = f"extract failed: {type(exc).__name__}: {exc}"
            return r
        except Exception as exc2:
            from pathlib import Path as _P
            p = _P(wav_path)
            return UtteranceRecord(
                basename=p.stem, rel_path=str(p), speaker_id=p.parent.name,
                sex=(p.parent.name[:1] if p.parent.name else ""), dialect_region="",
                split="", sample_rate=0, duration=0.0,
                config_hash=config.config_hash(), decode_ok=False,
                features={n: float("nan") for n in FEATURE_NAMES},
                coverage={n: 0 for n in FEATURE_NAMES},
                error=f"{exc}; recovery failed: {exc2}")


def extract_utterance(wav_path, timit_root, config: Config = CONFIG) -> UtteranceRecord:
    u = load_utterance(wav_path, timit_root)
    if not u.decode_ok or u.data is None:
        return _all_nan_record(u, config)

    fr = build_frames(u.n_samples, u.phones, config)

    perframe: dict = {}
    perframe.update(features_spectral.compute(u.data, config))
    pv = features_praat.compute(u.data, fr, u.phones, u.sex, config)
    perframe.update(pv)
    perframe.update(features_glottal.compute(u.data, fr, u.phones, u.sex, config))
    perframe.update(features_harmonic.compute(u.data, fr, pv["F0"], pv["CPP"], u.phones, config))
    perframe.update(features_formant.compute(u.data, fr, u.phones, u.sex, config))
    perframe["Nasality"] = features_nasality.compute(u.data, fr, perframe["F1"], config)
    perframe["VFI"] = deepfry_creak.compute(u.data, fr, config)   # DeepFry creak
    align = features_alignment.compute(u.phones, config)

    features: dict = {}
    coverage: dict = {}
    for spec in FEATURE_ORDER:
        if spec.level == "utterance":
            val, ncov = align[spec.name]
        else:
            val, ncov = aggregate_frame_feature(
                perframe[spec.name], fr.domain_mask(spec.domain), spec.aggregation, config)
        features[spec.name] = val
        coverage[spec.name] = ncov

    # VFI invariant: a voiced utterance ALWAYS has a defined fry proportion —
    # 0 when no creak is detected (incl. a DeepFry failure) — and is NaN only
    # when the utterance has no voiced frames. The min-valid-frames guard and
    # DeepFry crashes can otherwise leave a voiced utterance NaN; force it to 0.
    n_voiced = int(fr.voiced.sum())
    if not np.isfinite(features["VFI"]) and n_voiced > 0:
        features["VFI"] = 0.0
        coverage["VFI"] = n_voiced

    return UtteranceRecord(
        basename=u.basename, rel_path=u.rel_path, speaker_id=u.speaker_id, sex=u.sex,
        dialect_region=u.dialect_region, split=u.split, sample_rate=u.sample_rate,
        duration=u.duration, config_hash=config.config_hash(), decode_ok=True,
        features=features, coverage=coverage,
    )
