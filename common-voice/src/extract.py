"""Common Voice feature extraction (§B): MFA forced alignment → phone-class masks →
the same 40-feature extractor used for TIMIT (reused verbatim from voice-is-unique).

Run in the `voice-is-unique` env with the extractor on PYTHONPATH, e.g.:
    PYTHONPATH=~/claude/voice-is-unique/src python -m src.extract --config CONFIG/common_voice.json

Pipeline: decode mp3 → 16 kHz wav + .lab transcript → MFA align (english_us_arpa) →
TextGrid phones mapped to the TIMIT label space → voice-is-unique extractor → parquet.
MFA gives one segment per stop (no separate closure), so VOT (#39) degrades vs TIMIT
gold alignments — a known cross-corpus caveat. Multi-session CV ⇒ honest within-speaker
variance.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

CONDA = str(Path.home() / "miniconda3/bin/conda")


# ── phone mapping: MFA english_us_arpa (ARPABET+stress, sil/sp/spn) → TIMIT labels ──

def _phone_map(label: str, PHONE_CLASS) -> str:
    s = label.strip().lower().rstrip("0123456789")
    if s in ("", "sil", "sp", "spn", "<eps>", "spn"):
        return "h#"                                  # silence class
    return s if s in PHONE_CLASS else "h#"


def parse_textgrid(tg_path: str, sr: int, PHONE_CLASS):
    """Parse the MFA 'phones' tier → [Segment(start_sample, end_sample, timit_label)]."""
    from timit_features.io_timit import Segment
    text = Path(tg_path).read_text(encoding="utf-8")
    import re
    # locate the phones tier block (long-format TextGrid)
    blocks = re.split(r'item \[\d+\]:', text)
    phone_block = next((b for b in blocks if 'name = "phones"' in b), None)
    if phone_block is None:
        return []
    segs = []
    for m in re.finditer(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', phone_block):
        a, b, t = float(m.group(1)), float(m.group(2)), m.group(3)
        segs.append(Segment(int(round(a * sr)), int(round(b * sr)), _phone_map(t, PHONE_CLASS)))
    return segs


# ── audio + transcript prep for MFA ─────────────────────────────────────────────

def prep_corpus(man: pd.DataFrame, sent: dict, clips_dir: Path, corpus_dir: Path):
    """Decode each clip to 16 kHz wav and write its .lab transcript, grouped by speaker."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for r in man.itertuples():
        spk_dir = corpus_dir / r.speaker_id
        spk_dir.mkdir(exist_ok=True)
        stem = Path(r.clip_id).stem
        wav = spk_dir / f"{stem}.wav"
        if not wav.exists():
            subprocess.run(["ffmpeg", "-y", "-i", str(clips_dir / r.clip_id),
                            "-ac", "1", "-ar", "16000", str(wav)],
                           check=True, capture_output=True)
        (spk_dir / f"{stem}.lab").write_text(sent.get(r.clip_id, ""), encoding="utf-8")


def run_mfa(corpus_dir: Path, tg_dir: Path, cfg):
    m = cfg["masking"]
    subprocess.run([CONDA, "run", "-n", "aligner", "mfa", "align", "--clean", "-j", "4",
                    str(corpus_dir), m["mfa_dictionary"], m["mfa_acoustic_model"], str(tg_dir)],
                   check=True)


# ── feature extraction (reuse the voice-is-unique pipeline) ─────────────────────

def extract_clip(audio, phones, sex, cfg):
    """Replicates voice-is-unique extract_utterance on (audio, MFA phones)."""
    from timit_features.config import CONFIG, FEATURE_ORDER, FEATURE_NAMES
    from timit_features.framing import build_frames
    from timit_features.aggregate import aggregate_frame_feature
    from timit_features import (features_spectral, features_praat, features_glottal,
                                features_harmonic, features_formant, features_alignment,
                                features_nasality, deepfry_creak)
    fr = build_frames(len(audio), phones, CONFIG)
    pf = {}
    pf.update(features_spectral.compute(audio, CONFIG))
    pv = features_praat.compute(audio, fr, phones, sex, CONFIG); pf.update(pv)
    pf.update(features_glottal.compute(audio, fr, phones, sex, CONFIG))
    pf.update(features_harmonic.compute(audio, fr, pv["F0"], pv["CPP"], phones, CONFIG))
    pf.update(features_formant.compute(audio, fr, phones, sex, CONFIG))
    pf["Nasality"] = features_nasality.compute(audio, fr, pf["F1"], CONFIG)
    pf["VFI"] = deepfry_creak.compute(audio, fr, CONFIG)
    align = features_alignment.compute(phones, CONFIG)
    feats, cov = {}, {}
    for spec in FEATURE_ORDER:
        if spec.level == "utterance":
            v, n = align[spec.name]
        else:
            v, n = aggregate_frame_feature(pf[spec.name], fr.domain_mask(spec.domain),
                                           spec.aggregation, CONFIG)
        feats[spec.name], cov[spec.name] = v, n
    nv = int(fr.voiced.sum())
    if not np.isfinite(feats["VFI"]) and nv > 0:
        feats["VFI"], cov["VFI"] = 0.0, nv
    return feats, cov


def _decode_task(task):
    clip_path, wav_path, lab_path, sentence = task
    if not os.path.exists(wav_path):
        subprocess.run(["ffmpeg", "-y", "-i", clip_path, "-ac", "1", "-ar", "16000", wav_path],
                       check=True, capture_output=True)
    with open(lab_path, "w", encoding="utf-8") as f:
        f.write(sentence)


def _extract_task(task):
    from timit_features.config import PHONE_CLASS, FEATURE_NAMES
    spk, sex, lang, clip_id, dur, tg_path, wav_path, cfg = task
    rec = dict(speaker_id=spk, sex=sex, lang=lang, clip_id=clip_id,
               duration_ms=dur, decode_ok=False)
    try:
        audio, sr = sf.read(wav_path, dtype="float64")
        phones = parse_textgrid(tg_path, sr, PHONE_CLASS) if os.path.exists(tg_path) else []
        if phones:
            feats, cov = extract_clip(audio, phones, sex, cfg)
            rec.update({n: feats[n] for n in FEATURE_NAMES})
            rec.update({f"cov_{n}": cov[n] for n in FEATURE_NAMES})
            rec["decode_ok"] = True
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="CONFIG/common_voice.json")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    # keep per-worker BLAS/torch threads low so 4 workers don't oversubscribe 4 cores
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    cfg = json.loads(Path(args.config).expanduser().read_text())
    from timit_features.config import PHONE_CLASS, FEATURE_NAMES, CONFIG

    base = Path(cfg["raw_dir"]).expanduser() / args.lang
    clips_dir = base / "clips"
    man = pd.read_csv(base / "validated.tsv", sep="\t")
    man = man.rename(columns={"client_id": "speaker_id", "path": "clip_id"})
    if args.limit:
        man = man.head(args.limit)
    # sentences for MFA from the cached full train.tsv
    from huggingface_hub import hf_hub_download
    tr = pd.read_csv(hf_hub_download("fsicoli/common_voice_21_0", f"transcript/{args.lang}/train.tsv",
                                     repo_type="dataset"), sep="\t", usecols=["path", "sentence"],
                     quoting=3, dtype=str, on_bad_lines="skip")
    sent = dict(zip(tr.path, tr.sentence.fillna("")))

    import multiprocessing as mp
    corpus_dir = base / "_mfa_corpus"; tg_dir = base / "_mfa_aligned"

    print(f"[1/3] prep {len(man)} clips (decode + .lab, jobs={args.jobs}) ...", flush=True)
    dtasks = []
    for r in man.itertuples():
        spk_dir = corpus_dir / r.speaker_id; spk_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(r.clip_id).stem
        dtasks.append((str(clips_dir / r.clip_id), str(spk_dir / f"{stem}.wav"),
                       str(spk_dir / f"{stem}.lab"), sent.get(r.clip_id, "")))
    with mp.Pool(args.jobs) as pool:
        for i, _ in enumerate(pool.imap_unordered(_decode_task, dtasks, chunksize=16), 1):
            if i % 1000 == 0:
                print(f"  decoded {i}/{len(dtasks)}", flush=True)

    print("[2/3] MFA align (whole corpus, one invocation) ...", flush=True)
    run_mfa(corpus_dir, tg_dir, cfg)

    print(f"[3/3] extract features (jobs={args.jobs}) ...", flush=True)
    xtasks = []
    for r in man.itertuples():
        stem = Path(r.clip_id).stem
        xtasks.append((r.speaker_id, r.sex, args.lang, r.clip_id, r.duration_ms,
                       str(tg_dir / r.speaker_id / f"{stem}.TextGrid"),
                       str(corpus_dir / r.speaker_id / f"{stem}.wav"), cfg))
    rows = []
    with mp.Pool(args.jobs) as pool:
        for rec in pool.imap_unordered(_extract_task, xtasks, chunksize=4):
            rows.append(rec)
            if len(rows) % 200 == 0:
                print(f"  {len(rows)}/{len(xtasks)} (ok={sum(x['decode_ok'] for x in rows)})", flush=True)
    df = pd.DataFrame(rows)
    out = Path(cfg["feats_dir"]).expanduser(); out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "all_utterances.parquet", index=False)
    print(f"DONE: {df.decode_ok.sum()}/{len(df)} extracted → {out/'all_utterances.parquet'}", flush=True)


if __name__ == "__main__":
    main()
