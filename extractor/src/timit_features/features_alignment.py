"""Alignment-native features (computed from .PHN, not frame aggregation):
speech_rate (#38), VOT (#39), BGD (#40).

Each returns (value, n_events). Event-based validity (DECISIONS #8): NaN when no
qualifying event exists — never a frame-count guard.
"""
from __future__ import annotations

import numpy as np

from timit_features.config import Config, PHONE_CLASS
from timit_features.io_timit import Segment

_SILENCE_PHONES = ("h#", "pau", "epi")
_STOP_RELEASES = {"b", "d", "g", "p", "t", "k"}
_CLOSURES = {"bcl", "dcl", "gcl", "pcl", "tcl", "kcl"}


def _speaking_span(phones: list[Segment], sr: int) -> float:
    """Seconds from first to last non-silence segment (excludes lead/trail silence)."""
    speechy = [s for s in phones if s.label not in _SILENCE_PHONES]
    if not speechy:
        return 0.0
    return (speechy[-1].end - speechy[0].start) / sr


def speech_rate(phones: list[Segment], config: Config, sr: int) -> tuple[float, int]:
    """Syllables/sec; syllables = vowel-class nuclei (DECISIONS C3)."""
    nuclei = sum(1 for s in phones if PHONE_CLASS.get(s.label, "other") == "vowel")
    span = _speaking_span(phones, sr)
    if nuclei == 0 or span <= 0:
        return float("nan"), nuclei
    return nuclei / span, nuclei


def vot(phones: list[Segment], config: Config, sr: int) -> tuple[float, int]:
    """Mean VOT over stop releases. A release = a stop phone immediately preceded
    by a closure; VOT = release-segment duration. NaN if no release.
    NOTE: prevoiced (voiced) stops -> sign not captured here (flagged, DECISIONS #34)."""
    vots = []
    for i, s in enumerate(phones):
        if s.label in _STOP_RELEASES and i > 0 and phones[i - 1].label in _CLOSURES:
            vots.append((s.end - s.start) / sr)
    if not vots:
        return float("nan"), 0
    return float(np.mean(vots)), len(vots)


def bgd(phones: list[Segment], config: Config, sr: int) -> tuple[float, int]:
    """Breath-Group Duration: mean duration of maximal runs of contiguous
    non-silence between silence phones (h#/pau/epi). Closures do NOT break a run."""
    runs = []
    cur0 = cur1 = None
    for s in phones:
        if s.label in _SILENCE_PHONES:
            if cur0 is not None:
                runs.append((cur1 - cur0) / sr)
                cur0 = cur1 = None
        else:
            if cur0 is None:
                cur0 = s.start
            cur1 = s.end
    if cur0 is not None:
        runs.append((cur1 - cur0) / sr)
    if not runs:
        return float("nan"), 0
    return float(np.mean(runs)), len(runs)


def compute(phones: list[Segment], config: Config) -> dict[str, tuple[float, int]]:
    sr = config.framing.sample_rate_expected
    return {
        "speech_rate": speech_rate(phones, config, sr),
        "VOT": vot(phones, config, sr),
        "BGD": bgd(phones, config, sr),
    }
