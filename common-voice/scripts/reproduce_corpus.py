#!/usr/bin/env python
"""Recreate the Common Voice corpus from the committed MANIFEST: re-download every
clip, organize the gender-branched tree + speaker_map, and gender-classify the
"unknown" speakers by classifier confidence -- exactly as the corpus was built.

Inputs (committed, text only):
    manifest/corpus_clips.tsv      speaker_dir clip release split   (the file list)
    manifest/corpus_speakers.tsv   speaker_dir gender_dir client_id n_clips
                                    gender_source decision majority_count avg_conf

Phases (each resumable; re-run to continue after an interruption):
  1. download+organize  stream each release/split's audio tars, extract only the
        manifest's clips; write
            <out>/<rel_path>/<clip>.mp3            (gender-branched corpus tree)
            <out>/wavs/<speaker_dir>/<clip>.wav    (flat 16 kHz wav, for classify/MFA)
        and <out>/speaker_map.tsv. (.lab sentences written only with --align.)
  2. align (optional, --align)  MFA force-align -> <out>/phone_segments.parquet
        (needs the `aligner` conda env + sox; mirrors the original pipeline).
  3. classify  run gender_classify.py (env `gender-id`) over the unknown speakers'
        wavs -> <out>/unknown/estimated_gender, then VERIFY the freshly-estimated
        decisions against the frozen manifest and report any drift.

Run (env voice-is-unique; HF login required -- CV is gated):
    HF_TOKEN=...  python reproduce_corpus.py [--out DIR] [--workers 4] [--align]

The default --out is ~/data/commonvoice_repro so a reproduction never clobbers the
original ~/data/commonvoice. Point --out at a fresh dir for a clean rebuild.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
MANIFEST_DIR = HERE.parent / "manifest"
SR = 16000
GID_PY = os.environ.get("CV_GID_PY", os.path.expanduser("~/miniconda3/envs/gender-id/bin/python"))
# On PSC there is no ~/miniconda3; conda comes from `module load anaconda3`. Resolve from
# CV_CONDA, then PATH, then the laptop default.
CONDA = (os.environ.get("CV_CONDA") or shutil.which("conda")
         or os.path.expanduser("~/miniconda3/bin/conda"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _token():
    t = os.environ.get("HF_TOKEN")
    if not t:
        p = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(p):
            t = open(p).read().strip()
    if not t:
        sys.exit("HF_TOKEN not set and ~/.cache/huggingface/token missing (CV is gated).")
    return t


# ---------- manifest ----------
def read_manifest(mdir: Path):
    clips = list(csv.DictReader(open(mdir / "corpus_clips.tsv"), delimiter="\t"))
    spk = {r["speaker_dir"]: r
           for r in csv.DictReader(open(mdir / "corpus_speakers.tsv"), delimiter="\t")}
    return clips, spk


# ---------- phase 1: download + organize ----------
def load_sentences(rid: str, split: str) -> dict:
    """clip -> sentence, for .lab files (only needed for MFA alignment)."""
    tsv = hf_hub_download(rid, f"transcript/en/{split}.tsv", repo_type="dataset")
    import pandas as pd
    sent = {}
    for ch in pd.read_csv(tsv, sep="\t", usecols=["path", "sentence"], dtype=str,
                          quoting=csv.QUOTE_NONE, on_bad_lines="skip", chunksize=200_000):
        sent.update(dict(zip(ch["path"].fillna(""), ch["sentence"].fillna(""))))
    return sent


def load_sentences_local(cv_local, want: set) -> dict:
    """clip -> sentence from a LOCAL extracted CV release (<cv_local>/en/*.tsv).
    The HF 'train' split == the release's validated.tsv, so we scan the split TSVs
    and keep only the wanted clips; stop once every wanted clip is found."""
    import pandas as pd
    endir = Path(cv_local) / "en"
    sent = {}
    for name in ("validated.tsv", "other.tsv", "train.tsv", "invalidated.tsv"):
        p = endir / name
        if not p.exists():
            continue
        for ch in pd.read_csv(p, sep="\t", usecols=["path", "sentence"], dtype=str,
                              quoting=csv.QUOTE_NONE, on_bad_lines="skip", chunksize=200_000):
            for path, s in zip(ch["path"].fillna(""), ch["sentence"].fillna("")):
                if path in want:
                    sent[path] = s
        if len(sent) >= len(want):
            break
    return sent


def verify_local(clips, cv_local) -> list:
    """Fast presence check (NO copy/decode): confirm every manifest clip exists in a
    LOCAL extracted CV release's en/clips/. Reports counts broken down by source
    release (our clips span CV17/21/22, but CV22's clips/ is cumulative so all should
    be present); returns the list of any missing clips. Used by --verify-only."""
    clips_dir = Path(cv_local) / "en" / "clips"
    by_rel: dict[str, list] = collections.defaultdict(lambda: [0, 0])  # release -> [present, missing]
    missing = []
    for c in clips:
        present = (clips_dir / c["clip"]).exists()
        by_rel[c["release"]][0 if present else 1] += 1
        if not present:
            missing.append(c)
    n = len(clips)
    log(f"verify-only: {clips_dir}")
    log(f"verify-only: {n - len(missing)}/{n} clips present, {len(missing)} MISSING")
    for rel in sorted(by_rel):
        p, m = by_rel[rel]
        log(f"   {rel}: present={p} missing={m}")
    for c in missing[:30]:
        log(f"   MISSING {c['speaker_dir']} {c['clip']} [{c['release']}/{c['split']}]")
    if len(missing) > 30:
        log(f"   ... and {len(missing) - 30} more")
    return missing


def download_organize(clips, spk, out: Path, token: str, want_lab: bool, cv_local=None):
    out.mkdir(parents=True, exist_ok=True)
    (out / "wavs").mkdir(exist_ok=True)
    head = {"Authorization": "Bearer " + token}
    api = HfApi()

    # clip -> destinations
    mp3_path, wav_path, lab_path, clip_sd = {}, {}, {}, {}
    by_group: dict[tuple[str, str], set] = collections.defaultdict(set)
    for c in clips:
        sd, clip = c["speaker_dir"], c["clip"]
        rel = spk[sd]["gender_dir"]  # rel_path rebuilt below from speaker dir layout
        rel_path = _rel_path(sd, spk[sd]["gender_dir"])
        stem = clip[:-4]
        mp3_path[clip] = out / rel_path / clip
        wav_path[clip] = out / "wavs" / sd / f"{stem}.wav"
        lab_path[clip] = out / "wavs" / sd / f"{stem}.lab"
        clip_sd[clip] = sd
        if c["release"] != "UNRESOLVED":
            by_group[(c["release"], c["split"])].add(clip)

    def have(clip):
        return mp3_path[clip].exists() and wav_path[clip].exists()

    total_need = sum(1 for c in clips if not have(c["clip"]))
    log(f"download+organize: {len(clips)} clips, {total_need} still needed")

    def extract(clip, data, sentence):
        sd = clip_sd[clip]
        mp3_path[clip].parent.mkdir(parents=True, exist_ok=True)
        wav_path[clip].parent.mkdir(parents=True, exist_ok=True)
        mp3_path[clip].write_bytes(data)
        a, srr = sf.read(io.BytesIO(data))
        if a.ndim > 1:
            a = a.mean(1)
        if srr != SR:
            a = resample_poly(a, SR, srr)
        sf.write(wav_path[clip], a.astype(np.float32), SR, subtype="PCM_16")
        if want_lab:
            lab_path[clip].write_text(sentence, encoding="utf-8")

    # ---- LOCAL source: copy clips straight from an extracted CV release on disk ----
    if cv_local:
        clips_dir = Path(cv_local) / "en" / "clips"
        need = [c["clip"] for c in clips if not have(c["clip"])]
        sentences = load_sentences_local(cv_local, set(need)) if want_lab else {}
        log(f"local CV release {clips_dir}: {len(need)} clips to copy")
        missing = 0
        for i, clip in enumerate(need, 1):
            src = clips_dir / clip
            if not src.exists():
                missing += 1
                continue
            extract(clip, src.read_bytes(), sentences.get(clip, ""))
            if i % 5000 == 0:
                log(f"  ... {i}/{len(need)} copied")
        leftover = [c["clip"] for c in clips if not have(c["clip"])]
        log(f"local organize done; still-missing={len(leftover)} "
            f"(not present in release: {missing})")
        return leftover

    for (rid, split), group in sorted(by_group.items()):
        need = {c for c in group if not have(c)}
        if not need:
            continue
        sentences = load_sentences(rid, split) if want_lab else {}
        sib = [s.rfilename for s in api.repo_info(rid, repo_type="dataset").siblings]
        tars = sorted([f for f in sib if f.startswith(f"audio/en/{split}/") and f.endswith(".tar")],
                      key=lambda t: int(re.search(r'_(\d+)\.tar', t).group(1)), reverse=True)
        urlb = f"https://huggingface.co/datasets/{rid}/resolve/main/audio/en/{split}/"
        log(f"  {rid} {split}: {len(need)} clips across {len(tars)} shards")
        for t in tars:
            if not need:
                break
            name = os.path.basename(t)
            for attempt in range(6):
                try:
                    with requests.get(urlb + name, headers=head, stream=True, timeout=(30, 120)) as r:
                        r.raise_for_status()
                        with tarfile.open(fileobj=r.raw, mode="r|") as tf:
                            for mem in tf:
                                b = os.path.basename(mem.name)
                                if b in need:
                                    extract(b, tf.extractfile(mem).read(), sentences.get(b, ""))
                                    need.discard(b)
                    break
                except Exception as e:
                    log(f"    retry {name}: {type(e).__name__}"); time.sleep(5)
            log(f"    after {name}: {len(need)} {split}-clips still needed")

    leftover = [c["clip"] for c in clips if not have(c["clip"])]
    log(f"download done; still-missing={len(leftover)}")
    return leftover


def _rel_path(speaker_dir: str, gender_dir: str) -> str:
    # speaker_dir like 'f_1_1' / 'u_10_3' -> gender_dir/<prefix>_<grp>/<speaker_dir>
    prefix, grp, _idx = speaker_dir.split("_")
    return f"{gender_dir}/{prefix}_{grp}/{speaker_dir}"


def write_speaker_map(clips, spk, out: Path):
    by_sd = collections.defaultdict(list)
    for c in clips:
        by_sd[c["speaker_dir"]].append(c["clip"])
    rows = []
    for sd, s in spk.items():
        cl = sorted(by_sd.get(sd, []))
        rows.append([sd, _rel_path(sd, s["gender_dir"]), s["gender_dir"],
                     s["client_id"], s["n_clips"], ",".join(cl)])
    rows.sort(key=lambda r: r[0])
    with open(out / "speaker_map.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["speaker_dir", "rel_path", "gender_dir", "client_id", "n_clips", "clips"])
        w.writerows(rows)
    log(f"wrote {out/'speaker_map.tsv'} ({len(rows)} speakers)")


# ---------- phase 2: optional MFA alignment ----------
def align(out: Path):
    tg = out / "textgrids"
    tg.mkdir(exist_ok=True)
    log("MFA align (env aligner) -> textgrids ...")
    subprocess.run([CONDA, "run", "-n", "aligner", "mfa", "align", "--clean", "-j", "4",
                    str(out / "wavs"), "english_us_arpa", "english_us_arpa", str(tg)], check=True)
    # parse textgrids -> phone_segments.parquet
    import pandas as pd
    rows = []
    for tgf in tg.rglob("*.TextGrid"):
        sd = tgf.parent.name
        clip = tgf.stem + ".mp3"
        for ph, a, b in _parse_tg(tgf):
            rows.append((clip, sd, ph, a, b))
    pd.DataFrame(rows, columns=["clip_id", "speaker_dir", "phone", "start_s", "end_s"]) \
        .to_parquet(out / "phone_segments.parquet", index=False)
    log(f"wrote {out/'phone_segments.parquet'} ({len(rows)} phone rows)")


def _parse_tg(path):
    t = Path(path).read_text(encoding="utf-8")
    m = re.search(r'name = "phones".*?(?=item \[|\Z)', t, re.DOTALL)
    return [((mm.group(3) or "sil"), float(mm.group(1)), float(mm.group(2)))
            for mm in re.finditer(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"',
                                  m.group(0) if m else "")]


# ---------- phase 3: classify + verify ----------
def classify(out: Path, workers: int):
    log(f"classify unknown speakers (env gender-id, --workers {workers}) ...")
    (out / "unknown").mkdir(exist_ok=True)
    env = dict(os.environ, CV_ROOT=str(out), CV_WAVROOT=str(out / "wavs"))
    subprocess.run([GID_PY, str(HERE / "gender_classify.py"),
                    "--workers", str(workers), "--threads", "1"], check=False, env=env)


def verify(spk, out: Path):
    est_f = out / "unknown" / "estimated_gender"
    if not est_f.exists():
        log("no estimated_gender produced; skipping verification"); return
    fresh = {r["speaker_dir"]: r["decision"]
             for r in csv.DictReader(open(est_f), delimiter="\t")}
    n = agree = drift = 0
    examples = []
    for sd, s in spk.items():
        if s["gender_source"] != "estimated":
            continue
        n += 1
        want, got = s["decision"], fresh.get(sd)
        if got is None:
            continue
        if got == want:
            agree += 1
        else:
            drift += 1
            if len(examples) < 10:
                examples.append(f"{sd}: manifest={want} reproduced={got}")
    log(f"VERIFY estimated decisions: {agree}/{n} match manifest; drift={drift}")
    for e in examples:
        log("   " + e)
    if drift:
        log("   (small drift is expected -- classifier is deterministic per clip but a few "
            "borderline speakers can flip; re-downloaded audio is byte-identical so drift "
            "should be near zero.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default=str(MANIFEST_DIR))
    ap.add_argument("--out", default=os.path.expanduser("~/data/commonvoice_repro"))
    ap.add_argument("--workers", type=int, default=4,
                    help="classifier workers (4 avoids the documented 8-worker OOM stall)")
    ap.add_argument("--align", action="store_true", help="also MFA-align -> phone_segments.parquet")
    ap.add_argument("--skip-classify", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="rebuild only the first N speakers (smoke test / partial rebuild)")
    ap.add_argument("--cv-local", default=os.environ.get("CV_LOCAL"),
                    help="path to an extracted CV release (dir containing en/clips + en/*.tsv); "
                         "copy clips from there instead of downloading from HuggingFace")
    ap.add_argument("--verify-only", action="store_true",
                    help="with --cv-local: only check that every manifest clip exists in the "
                         "release's en/clips/ (no copy/decode/align); exit nonzero if any missing")
    args = ap.parse_args()

    out = Path(args.out)
    cv_local = args.cv_local
    clips, spk = read_manifest(Path(args.manifest_dir))
    if args.limit:
        keep = set(list(spk)[:args.limit])
        spk = {sd: r for sd, r in spk.items() if sd in keep}
        clips = [c for c in clips if c["speaker_dir"] in keep]
    src = f"local:{cv_local}" if cv_local else "HuggingFace"
    log(f"manifest: {len(spk)} speakers, {len(clips)} clips -> out={out}  (source: {src})")

    if args.verify_only:
        if not cv_local:
            sys.exit("--verify-only requires --cv-local (or CV_LOCAL)")
        missing = verify_local(clips, cv_local)
        sys.exit(1 if missing else 0)

    token = "" if cv_local else _token()
    download_organize(clips, spk, out, token, want_lab=args.align, cv_local=cv_local)
    write_speaker_map(clips, spk, out)
    if args.align:
        align(out)
    if not args.skip_classify:
        classify(out, args.workers)
        verify(spk, out)
    log("=== reproduce_corpus done ===")


if __name__ == "__main__":
    main()
