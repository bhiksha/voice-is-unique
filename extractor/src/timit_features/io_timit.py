"""TIMIT I/O: decode SPHERE audio and parse the alignment/transcript files.

The TIMIT corpus is treated as strictly read-only. A decode failure is reported
(not raised) so the caller can mark all features NaN and continue.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Segment:
    """One alignment interval: [start, end) in samples, with a label."""
    start: int
    end: int
    label: str


@dataclass
class Utterance:
    """A loaded TIMIT utterance: audio + alignments + identifiers."""
    # identifiers
    basename: str
    rel_path: str            # path relative to the TIMIT root, e.g. TRAIN/DR1/FCJF0/SA1.WAV
    split: str               # TRAIN | TEST
    dialect_region: str      # DR1..DR8
    speaker_id: str          # e.g. FCJF0
    sex: str                 # M | F  (first char of speaker_id)
    # audio
    sample_rate: int
    data: np.ndarray | None  # mono float64, or None if decode failed
    n_samples: int
    duration: float
    # alignments / transcript
    phones: list[Segment]    # from .PHN
    words: list[Segment]     # from .WRD
    text: str                # from .TXT (orthographic transcript)
    # status
    decode_ok: bool
    error: str | None = None


def _parse_alignment(path: Path) -> list[Segment]:
    """Parse a TIMIT .PHN/.WRD file: lines of `start_sample end_sample label`."""
    segs: list[Segment] = []
    if not path.exists():
        return segs
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            segs.append(Segment(int(parts[0]), int(parts[1]), parts[2]))
    return segs


def _parse_text(path: Path) -> str:
    """Parse a TIMIT .TXT file: `start end <orthographic sentence>`."""
    if not path.exists():
        return ""
    line = path.read_text(encoding="utf-8").strip()
    parts = line.split(maxsplit=2)
    return parts[2] if len(parts) >= 3 else line


def _identify(wav_path: Path, timit_root: Path) -> dict:
    """Derive identifiers from the standard TIMIT path layout."""
    rel = wav_path.relative_to(timit_root)
    speaker_id = wav_path.parent.name
    return {
        "basename": wav_path.stem,
        "rel_path": str(rel),
        "split": rel.parts[0] if len(rel.parts) > 0 else "",
        "dialect_region": rel.parts[1] if len(rel.parts) > 1 else "",
        "speaker_id": speaker_id,
        "sex": speaker_id[0].upper() if speaker_id else "",
    }


def load_utterance(wav_path: str | Path, timit_root: str | Path) -> Utterance:
    """Load one TIMIT utterance (audio + alignments + ids). Never raises on a
    decode failure — sets decode_ok=False and data=None instead."""
    wav_path = Path(wav_path)
    timit_root = Path(timit_root)
    ids = _identify(wav_path, timit_root)

    base = wav_path.with_suffix("")
    phones = _parse_alignment(base.with_suffix(".PHN"))
    words = _parse_alignment(base.with_suffix(".WRD"))
    text = _parse_text(base.with_suffix(".TXT"))

    data: np.ndarray | None = None
    sr = 0
    decode_ok = True
    error = None
    try:
        data, sr = sf.read(str(wav_path), dtype="float64", always_2d=False)
        if data.ndim > 1:                      # collapse any multichannel to mono
            data = data.mean(axis=1)
    except Exception as exc:                    # decode failure → mark, don't crash
        decode_ok = False
        error = f"{type(exc).__name__}: {exc}"

    n = int(data.shape[0]) if data is not None else 0
    return Utterance(
        sample_rate=int(sr),
        data=data,
        n_samples=n,
        duration=(n / sr) if sr else 0.0,
        phones=phones,
        words=words,
        text=text,
        decode_ok=decode_ok,
        error=error,
        **ids,
    )
