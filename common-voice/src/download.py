"""Common Voice acquisition + speaker/clip selection (§A), deterministic and seeded.

Selects, from a Common Voice release, the speakers with >= clips_per_speaker clips
of duration >= clip_min_sec, takes n_per_sex male + n_per_sex female (binary self-
reported gender only), exactly clips_per_speaker clips each (first by sorted clip
id), and writes a frozen clip-ID manifest. The MANIFEST (not audio) is committed.

Common Voice 21.0 is gated: you must accept the dataset terms with your Hugging
Face account and be logged in (`huggingface-cli login`) before this can stream.
This module streams metadata to select, then fetches only the selected clips'
audio to ~/data/commonvoice (never the whole release).

NOTE: CV's HF schema across releases exposes per-clip `client_id`, `gender`,
`sentence`, and audio; clip duration is read from the decoded audio or a supplied
`clip_durations.tsv`. The exact HF repo id for the release is in CONFIG
(`cv_release`). Verify availability before a full run; if the release is website-
only, point --metadata-tsv at the release's `validated.tsv` instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# CV gender labels → binary M/F (schema has varied across releases).
_MALE = {"male", "male_masculine"}
_FEMALE = {"female", "female_feminine"}


def map_gender(label) -> str | None:
    if not isinstance(label, str):
        return None
    g = label.strip().lower()
    if g in _MALE:
        return "M"
    if g in _FEMALE:
        return "F"
    return None                       # missing / other / non-binary → excluded (CONFIG)


def select_speakers(meta: pd.DataFrame, cfg) -> pd.DataFrame:
    """meta: per-clip rows with columns [client_id, gender, clip_id, duration_sec].
    Returns the frozen manifest (speaker_id, sex, clip_id, duration_sec)."""
    sel = cfg["selection"]
    meta = meta.copy()
    meta["sex"] = meta["gender"].map(map_gender)
    meta = meta[meta["sex"].notna()]
    meta = meta[meta["duration_sec"] >= sel["clip_min_sec"]]

    # qualifying speakers: >= clips_per_speaker long-enough clips
    counts = meta.groupby("client_id").size()
    qualifying = counts[counts >= sel["clips_per_speaker"]].index
    meta = meta[meta["client_id"].isin(qualifying)]

    rows = []
    avail = {}
    for sex in ("M", "F"):
        spk = sorted(meta[meta["sex"] == sex]["client_id"].unique())
        avail[sex] = len(spk)
        rng = np.random.default_rng(sel["seed"])
        spk = list(np.array(spk)[rng.permutation(len(spk))])[: sel["n_per_sex"]]
        for s in spk:
            clips = sorted(meta[meta["client_id"] == s]["clip_id"].tolist())[: sel["clips_per_speaker"]]
            for c in clips:
                d = float(meta[(meta["client_id"] == s) & (meta["clip_id"] == c)]["duration_sec"].iloc[0])
                rows.append(dict(speaker_id=s, sex=sex, clip_id=c, duration_sec=d))
    man = pd.DataFrame(rows)
    man.attrs["available_per_sex"] = avail
    return man


def report_availability(man: pd.DataFrame, cfg):
    avail = man.attrs.get("available_per_sex", {})
    need = cfg["selection"]["n_per_sex"]
    print(f"qualifying speakers: M={avail.get('M')} F={avail.get('F')} (need {need} each)")
    for sex in ("M", "F"):
        if avail.get(sex, 0) < need:
            print(f"  !! only {avail.get(sex)} {sex} speakers qualify (< {need}) — STOP and report.")
    got = man.groupby("speaker_id").size()
    print(f"selected: {man['speaker_id'].nunique()} speakers, "
          f"{len(man)} clips; clips/speaker min={got.min()} max={got.max()}; "
          f"all >= {cfg['selection']['clip_min_sec']}s: {bool((man['duration_sec'] >= cfg['selection']['clip_min_sec']).all())}")


def write_manifest(man: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(path, index=False)
    meta = dict(n_speakers=int(man["speaker_id"].nunique()), n_clips=int(len(man)),
                by_sex={s: int(man[man["sex"] == s]["speaker_id"].nunique()) for s in ("M", "F")})
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Common Voice selection + manifest")
    ap.add_argument("--config", default="CONFIG/common_voice.json")
    ap.add_argument("--metadata-tsv", required=True,
                    help="per-clip metadata with client_id/gender/path/duration "
                         "(release validated.tsv, or a streamed-and-cached export)")
    ap.add_argument("--pilot", action="store_true", help="use pilot n_per_sex if set in CONFIG")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).expanduser().read_text())

    meta = pd.read_csv(args.metadata_tsv, sep="\t")
    # normalise columns to [client_id, gender, clip_id, duration_sec]
    meta = meta.rename(columns={"path": "clip_id"})
    if "duration_sec" not in meta.columns and "duration" in meta.columns:
        meta["duration_sec"] = meta["duration"] / 1000.0 if meta["duration"].max() > 1000 else meta["duration"]
    man = select_speakers(meta, cfg)
    report_availability(man, cfg)
    write_manifest(man, Path(cfg["manifest"]))
    print(f"manifest → {cfg['manifest']}")


if __name__ == "__main__":
    main()
