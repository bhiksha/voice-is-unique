"""Fixed-grid framing and the .PHN-driven phone-class / domain masks.

Each frame is mapped to a phone by its CENTER sample (CONFIG.framing.frame_to_phone
== "center_sample"), then to a phone class, then to the per-feature domains
(voiced / sonorant / speech / sibilant). Silence and stop closures are excluded.
No energy-based VAD is ever used — masking comes only from the alignment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from timit_features.config import (
    Config, PHONE_CLASS, VOICED_CLASSES, SONORANT_CLASSES,
    SPEECH_CLASSES, SILENCE_CLASSES, SIBILANT_PHONES,
)
from timit_features.io_timit import Segment


@dataclass
class Frames:
    """Per-frame grid and masks for one utterance."""
    center_samples: np.ndarray   # (n_frames,) int — center sample of each frame
    start_samples: np.ndarray    # (n_frames,) int — first sample of each frame
    frame_length: int            # samples
    hop: int                     # samples
    phone: list[str]             # (n_frames,) ARPABET label at the frame center
    phone_class: np.ndarray      # (n_frames,) object — phone class per frame
    voiced: np.ndarray           # (n_frames,) bool
    sonorant: np.ndarray         # bool
    speech: np.ndarray           # bool
    sibilant: np.ndarray         # bool

    @property
    def n_frames(self) -> int:
        return len(self.center_samples)

    def domain_mask(self, domain: str) -> np.ndarray:
        return {
            "voiced": self.voiced, "sonorant": self.sonorant,
            "speech": self.speech, "sibilant": self.sibilant,
        }[domain]


def _phone_at(segments: list[Segment], sample: int) -> str:
    """Return the phone label whose [start, end) contains `sample` (or '' )."""
    # TIMIT alignments are contiguous and ordered; a linear scan is fine per frame,
    # but we binary-search the starts for speed and determinism.
    if not segments:
        return ""
    lo, hi = 0, len(segments) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s = segments[mid]
        if sample < s.start:
            hi = mid - 1
        elif sample >= s.end:
            lo = mid + 1
        else:
            return s.label
    return ""


def build_frames(n_samples: int, phones: list[Segment], config: Config) -> Frames:
    """Build the frame grid and all phone-class/domain masks for an utterance."""
    fr = config.framing
    sr = fr.sample_rate_expected
    frame_length = int(round(fr.frame_length_ms * 1e-3 * sr))
    hop = int(round(fr.hop_ms * 1e-3 * sr))

    # Frames fully inside the signal: start at i*hop, length frame_length.
    if n_samples >= frame_length:
        n_frames = 1 + (n_samples - frame_length) // hop
    else:
        n_frames = 0
    starts = np.arange(n_frames, dtype=np.int64) * hop
    centers = starts + frame_length // 2

    labels = [_phone_at(phones, int(c)) for c in centers]
    classes = np.array([PHONE_CLASS.get(p, "other") for p in labels], dtype=object)

    voiced_set, son_set = set(VOICED_CLASSES), set(SONORANT_CLASSES)
    speech_set = set(SPEECH_CLASSES)
    sib_set = set(SIBILANT_PHONES)

    voiced = np.array([c in voiced_set for c in classes], dtype=bool)
    sonorant = np.array([c in son_set for c in classes], dtype=bool)
    speech = np.array([c in speech_set for c in classes], dtype=bool)
    sibilant = np.array([p in sib_set for p in labels], dtype=bool)

    return Frames(
        center_samples=centers, start_samples=starts,
        frame_length=frame_length, hop=hop,
        phone=labels, phone_class=classes,
        voiced=voiced, sonorant=sonorant, speech=speech, sibilant=sibilant,
    )
