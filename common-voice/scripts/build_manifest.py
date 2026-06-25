#!/usr/bin/env python
"""Freeze the built Common Voice corpus into a committable, re-downloadable MANIFEST.

This reads the LIVE corpus produced by the ~/data pipeline
(`commonvoice/speaker_map.tsv` + `unknown/estimated_gender`) and writes two small
TSVs that, together, are a complete and reproducible record of "all the downloaded
files" plus how each speaker's gender was decided:

  manifest/corpus_speakers.tsv   one row / speaker
      speaker_dir gender_dir client_id n_clips gender_source decision
      majority_count avg_conf
        - gender_source: self_reported (female/ or male/ dir) | estimated (unknown/ dir)
        - for estimated speakers, decision/majority_count/avg_conf come from
          unknown/estimated_gender (wav2vec2 classifier; >=9/10 -> tagged_<g>).

  manifest/corpus_clips.tsv      one row / clip  (the list of all downloaded files)
      speaker_dir clip release split
        - release: the HF dataset id the clip's audio is fetched from
          (e.g. fsicoli/common_voice_22_0); split in {train, other}.
        - resolved NEWEST-first against the cached CV metadata, because CV is
          cumulative: a clip still present in CV22 is fetched from CV22.

Only the manifest (text) is committed; the audio is re-fetched by reproduce_corpus.py.

Run (env voice-is-unique, with the CV metadata TSVs already in the HF cache):
    HF_TOKEN=... python build_manifest.py [--corpus ~/data/commonvoice] [--out ../manifest]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

# CV releases to resolve clips against, NEWEST first (cumulative -> prefer latest).
RELEASES = [
    "fsicoli/common_voice_22_0",
    "fsicoli/common_voice_21_0",
    "fsicoli/common_voice_17_0",
]
SPLITS = ["train", "other"]  # the only splits whose audio tars the pipeline fetches


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_split_paths(rid: str, split: str) -> set[str]:
    """Set of clip filenames present in `rid`'s `split`, from the cached metadata TSV."""
    tsv = hf_hub_download(rid, f"transcript/en/{split}.tsv", repo_type="dataset")
    paths: set[str] = set()
    # read only the `path` column, in chunks, to keep memory modest on 358 MB files
    for chunk in pd.read_csv(tsv, sep="\t", usecols=["path"], dtype=str,
                             quoting=csv.QUOTE_NONE, on_bad_lines="skip",
                             chunksize=200_000):
        paths.update(chunk["path"].dropna().tolist())
    return paths


def read_speaker_map(corpus: Path):
    rows = list(csv.DictReader(open(corpus / "speaker_map.tsv"), delimiter="\t"))
    speakers = []
    for r in rows:
        clips = [c for c in r["clips"].split(",") if c]
        speakers.append({
            "speaker_dir": r["speaker_dir"],
            "gender_dir": r["gender_dir"],
            "client_id": r["client_id"],
            "n_clips": r["n_clips"],
            "clips": clips,
        })
    return speakers


def read_estimated_gender(corpus: Path):
    f = corpus / "unknown" / "estimated_gender"
    if not f.exists():
        return {}
    est = {}
    for r in csv.DictReader(open(f), delimiter="\t"):
        est[r["speaker_dir"]] = r
    return est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.expanduser("~/data/commonvoice"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "manifest"))
    args = ap.parse_args()

    corpus, out = Path(args.corpus), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    speakers = read_speaker_map(corpus)
    est = read_estimated_gender(corpus)
    log(f"corpus: {len(speakers)} speakers, {sum(len(s['clips']) for s in speakers)} clips")

    # ---- resolve every clip -> (release, split), newest release first ----
    wanted = {c for s in speakers for c in s["clips"]}
    resolved: dict[str, tuple[str, str]] = {}
    for rid in RELEASES:
        for split in SPLITS:
            if not wanted:
                break
            paths = load_split_paths(rid, split)
            hit = wanted & paths
            for c in hit:
                resolved[c] = (rid, split)
            wanted -= hit
            log(f"  {rid} {split}: resolved {len(hit)} (remaining {len(wanted)})")
        if not wanted:
            break
    if wanted:
        log(f"WARNING: {len(wanted)} clips not found in any release/split "
            f"(audio may have been re-validated/removed). Listed with release=UNRESOLVED.")

    # ---- write corpus_clips.tsv ----
    clips_path = out / "corpus_clips.tsv"
    with open(clips_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["speaker_dir", "clip", "release", "split"])
        for s in speakers:
            for c in s["clips"]:
                rid, split = resolved.get(c, ("UNRESOLVED", ""))
                w.writerow([s["speaker_dir"], c, rid, split])
    log(f"wrote {clips_path}")

    # ---- write corpus_speakers.tsv ----
    spk_path = out / "corpus_speakers.tsv"
    with open(spk_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["speaker_dir", "gender_dir", "client_id", "n_clips",
                    "gender_source", "decision", "majority_count", "avg_conf"])
        for s in speakers:
            if s["gender_dir"] in ("female", "male"):
                w.writerow([s["speaker_dir"], s["gender_dir"], s["client_id"],
                            s["n_clips"], "self_reported", s["gender_dir"], "", ""])
            else:  # unknown -> estimated
                e = est.get(s["speaker_dir"], {})
                w.writerow([s["speaker_dir"], s["gender_dir"], s["client_id"],
                            s["n_clips"], "estimated",
                            e.get("decision", ""), e.get("majority_count", ""),
                            e.get("avg_conf", "")])
    log(f"wrote {spk_path}")

    # ---- console summary ----
    by_rel = {}
    for rid, _ in resolved.values():
        by_rel[rid] = by_rel.get(rid, 0) + 1
    log("clips per release: " + ", ".join(f"{k}={v}" for k, v in sorted(by_rel.items())))
    if wanted:
        log(f"UNRESOLVED clips: {len(wanted)}")
        sys.exit(2 if len(wanted) > 0.001 * sum(len(s['clips']) for s in speakers) else 0)


if __name__ == "__main__":
    main()
