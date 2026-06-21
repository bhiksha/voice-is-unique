"""Output writers: per-utterance JSON (mirroring TIMIT layout), the consolidated
all_utterances table (.parquet + .csv), CONFIG.json, and MANIFEST.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from timit_features.config import Config, FEATURE_NAMES
from timit_features.extract import UtteranceRecord

_ID_COLS = ["basename", "rel_path", "speaker_id", "sex", "dialect_region",
            "split", "sample_rate", "duration", "decode_ok", "config_hash"]
_COV_COLS = [f"cov_{n}" for n in FEATURE_NAMES]
COLUMNS = _ID_COLS + list(FEATURE_NAMES) + _COV_COLS   # fixed, documented order


def record_to_dict(r: UtteranceRecord) -> dict:
    d = {c: getattr(r, c) for c in _ID_COLS}
    d["features"] = {n: r.features[n] for n in FEATURE_NAMES}
    d["coverage"] = {n: r.coverage[n] for n in FEATURE_NAMES}
    if r.error:
        d["error"] = r.error
    return d


def write_utterance_json(r: UtteranceRecord, out_root: Path) -> Path:
    """Write <out_root>/<rel_path without ext>.json, mirroring TIMIT structure."""
    rel = Path(r.rel_path).with_suffix(".json")
    path = Path(out_root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record_to_dict(r), indent=2), encoding="utf-8")
    return path


def _row(r: UtteranceRecord) -> dict:
    row = {c: getattr(r, c) for c in _ID_COLS}
    row.update({n: r.features[n] for n in FEATURE_NAMES})
    row.update({f"cov_{n}": r.coverage[n] for n in FEATURE_NAMES})
    return row


def build_dataframe(records: list[UtteranceRecord]) -> pd.DataFrame:
    df = pd.DataFrame([_row(r) for r in records], columns=COLUMNS)
    return df.sort_values("rel_path").reset_index(drop=True)


def write_table(df: pd.DataFrame, out_root: Path) -> None:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_root / "all_utterances.parquet", index=False)
    df.to_csv(out_root / "all_utterances.csv", index=False)


def read_records(out_root: Path) -> list[UtteranceRecord]:
    """Reconstruct UtteranceRecords from every per-utterance JSON under out_root
    (so the table/manifest reflect ALL completed work, including prior runs)."""
    out_root = Path(out_root)
    recs: list[UtteranceRecord] = []
    for jp in sorted(out_root.rglob("*.json")):
        if jp.name in ("CONFIG.json", "MANIFEST.json"):
            continue
        d = json.loads(jp.read_text(encoding="utf-8"))
        if "basename" not in d or "features" not in d:
            continue                                   # skip non-record JSONs (e.g. provenance)
        recs.append(UtteranceRecord(
            basename=d["basename"], rel_path=d["rel_path"], speaker_id=d["speaker_id"],
            sex=d["sex"], dialect_region=d["dialect_region"], split=d["split"],
            sample_rate=d["sample_rate"], duration=d["duration"],
            config_hash=d["config_hash"], decode_ok=d["decode_ok"],
            features={n: d["features"][n] for n in FEATURE_NAMES},
            coverage={n: d["coverage"][n] for n in FEATURE_NAMES},
            error=d.get("error")))
    return recs


def write_config(config: Config, out_root: Path) -> None:
    (Path(out_root) / "CONFIG.json").write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True), encoding="utf-8")


def write_manifest(records: list[UtteranceRecord], config: Config, out_root: Path) -> None:
    seen = len(records)
    decoded = sum(1 for r in records if r.decode_ok)
    failed = seen - decoded
    # per-feature coverage = fraction of utterances with a non-NaN value
    cov = {}
    for n in FEATURE_NAMES:
        vals = np.array([r.features[n] for r in records], dtype=np.float64)
        cov[n] = float(np.isfinite(vals).mean()) if seen else 0.0
    spk_counts: dict[str, int] = {}
    for r in records:
        spk_counts[r.speaker_id] = spk_counts.get(r.speaker_id, 0) + 1
    manifest = {
        "config_hash": config.config_hash(),
        "utterances_seen": seen,
        "utterances_decoded": decoded,
        "utterances_failed_decode": failed,
        "decode_failures": [r.rel_path for r in records if not r.decode_ok],
        "n_speakers": len(spk_counts),
        "per_speaker_utterance_counts": dict(sorted(spk_counts.items())),
        "per_feature_noNaN_fraction": cov,
    }
    (Path(out_root) / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
