#!/usr/bin/env python
"""Compute the 40-feature acoustic representation for a balanced 4,000-male +
4,000-female subset of the built Common Voice corpus, into a PARALLEL directory
tree under commonvoice-feats/ (mirroring the corpus speaker tree).

This is a long, SUSTAINED, RESUMABLE job: each clip's DeepFormants (F1-F4) and
DeepFry (VFI) steps spawn one model-load subprocess apiece (~seconds/clip), so the
full ~80,000-clip run takes many hours. Every clip's result is written atomically as
its own JSON the instant it is computed; on (re)start, clips whose JSON already
exists are skipped -- so a crash costs at most the in-flight clips. Re-run the exact
same command to resume.

Selection (deterministic): a speaker's sex is self-reported (female/male dirs) or, for
"unknown" speakers, the classifier tag (tagged_female->F, tagged_male->M; ambiguous/
tie excluded). Within each sex the speakers are sorted by natural speaker_dir key and
the first N (=4,000) taken.

Inputs (already built; NOT re-derived here):
  --corpus  ~/data/commonvoice        speaker_map.tsv + unknown/estimated_gender
            ~/data/commonvoice/phone_segments.parquet   MFA phones (all speakers)
  --wavroot ~/cv_align/corpus         per-speaker 16 kHz wavs (<sd>/<stem>.wav)
Output:
  --out     ~/data/commonvoice-feats  <gender>/<grp>/<sd>/<stem>.json  (+ all_utterances.parquet)
  manifest  committed TSV of the 8,000 selected speakers.

The 40-feature extractor is reused VERBATIM from voice-is-unique (extract_clip); no
methodology is changed here. Run in the `voice-is-unique` env (it shells out to the
`deepformants` and `deepfry` envs itself):

    python extract_feats_8k.py [--jobs 6] [--n-per-sex 4000]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf

HOME = Path.home()
# Resolve code paths RELATIVE TO THE REPO so this runs unchanged anywhere (laptop, PSC, ...).
# This file lives at <repo>/common-voice/scripts/extract_feats_8k.py:
CV_PKG = Path(os.environ.get("CV_PKG", Path(__file__).resolve().parent.parent))   # <repo>/common-voice
EXTRACTOR_SRC = Path(os.environ.get(                                              # <repo>/extractor/src
    "EXTRACTOR_SRC", CV_PKG.parent / "extractor" / "src"))
if not (EXTRACTOR_SRC / "timit_features").is_dir():                               # fallback: dev-repo layout
    EXTRACTOR_SRC = HOME / "claude/voice-is-unique/src"
sys.path.insert(0, str(EXTRACTOR_SRC))
sys.path.insert(0, str(CV_PKG))

DEF_CORPUS = HOME / "data/commonvoice"
DEF_WAVROOT = HOME / "cv_align/corpus"
DEF_OUT = HOME / "data/commonvoice-feats"
DEF_MANIFEST = CV_PKG / "manifest" / "feats_manifest_8000.tsv"


DEF_CLAUDEMD = HOME / "data" / "CLAUDE.md"
PROG_MARK = "- **LIVE PROGRESS"   # the line in CLAUDE.md this job rewrites every 1000 clips


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def update_claudemd(path, done, total, ok, rate, eta):
    """Rewrite the single PROG_MARK line in CLAUDE.md (crash-recovery status). No-op if
    the file or marker line is absent; never appends. Atomic replace."""
    try:
        p = Path(path)
        if not p.exists():
            return
        new = (f"{PROG_MARK} (auto-updated every 1000 clips):** "
               f"done={done}/{total} ok={ok} rate={rate:.2f}/s ETA={eta:.1f}h "
               f"@{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines = p.read_text().splitlines(keepends=True)
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith(PROG_MARK):
                lines[i] = new
                break
        else:
            return  # marker not present — do not blindly append
        tmp = str(p) + ".tmp"
        Path(tmp).write_text("".join(lines))
        os.replace(tmp, p)
    except Exception:
        pass


def natkey(sd: str):
    p, g, i = sd.split("_")
    return (p, int(g), int(i))


# ── selection ────────────────────────────────────────────────────────────────
def sex_of(row, est):
    gd = row["gender_dir"]
    if gd == "female":
        return "F", "self_reported", "female"
    if gd == "male":
        return "M", "self_reported", "male"
    e = est.get(row["speaker_dir"])
    if e:
        d = e.get("decision", "")
        if d == "tagged_female":
            return "F", "estimated", d
        if d == "tagged_male":
            return "M", "estimated", d
    return None, None, None  # ambiguous / tie / unknown -> excluded


def select(corpus: Path, n_per_sex: int):
    spk = list(csv.DictReader(open(corpus / "speaker_map.tsv"), delimiter="\t"))
    estf = corpus / "unknown" / "estimated_gender"
    est = {r["speaker_dir"]: r for r in csv.DictReader(open(estf), delimiter="\t")} if estf.exists() else {}
    pools = {"F": [], "M": []}
    for r in spk:
        sx, src, dec = sex_of(r, est)
        if sx:
            pools[sx].append((r, src, dec))
    chosen = []
    for sx in ("F", "M"):
        pools[sx].sort(key=lambda t: natkey(t[0]["speaker_dir"]))
        for (r, src, dec) in pools[sx][:n_per_sex]:
            chosen.append({"speaker_dir": r["speaker_dir"], "sex": sx, "gender_source": src,
                           "decision": dec, "rel_path": r["rel_path"],
                           "client_id": r["client_id"], "n_clips": r["n_clips"],
                           "clips": r["clips"]})
    return chosen, len(pools["F"]), len(pools["M"])


def load_chosen_manifest(path: Path):
    """Use a FROZEN feats manifest (speaker_dir, sex, ..., rel_path, clips) verbatim
    instead of re-selecting/re-classifying — guarantees the exact study speakers + sex
    labels for repeatability (no classifier drift)."""
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    chosen = [{"speaker_dir": r["speaker_dir"], "sex": r["sex"],
               "gender_source": r.get("gender_source", ""), "decision": r.get("decision", ""),
               "rel_path": r["rel_path"], "client_id": r.get("client_id", ""),
               "n_clips": r.get("n_clips", ""), "clips": r["clips"]} for r in rows]
    nF = sum(c["sex"] == "F" for c in chosen)
    nM = sum(c["sex"] == "M" for c in chosen)
    return chosen, nF, nM


def write_manifest(chosen, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["speaker_dir", "sex", "gender_source", "decision", "rel_path",
                    "client_id", "n_clips", "clips"])
        for c in chosen:
            w.writerow([c["speaker_dir"], c["sex"], c["gender_source"], c["decision"],
                        c["rel_path"], c["client_id"], c["n_clips"], c["clips"]])


# ── per-clip extraction (worker) ──────────────────────────────────────────────
def _extract_one(task):
    from src.extract import extract_clip, _phone_map
    from timit_features.config import CONFIG, PHONE_CLASS, FEATURE_NAMES
    from timit_features.io_timit import Segment
    sd, sex, clip_id, phrows, wav_path, out_json = task
    rec = dict(speaker_id=sd, sex=sex, lang="en", clip_id=clip_id, decode_ok=False)
    try:
        audio, sr = sf.read(wav_path, dtype="float64")
        phones = [Segment(int(round(s * sr)), int(round(e * sr)), _phone_map(ph, PHONE_CLASS))
                  for ph, s, e in phrows]
        if phones:
            feats, cov = extract_clip(audio, phones, sex, CONFIG)
            rec.update({n: feats[n] for n in FEATURE_NAMES})
            rec.update({f"cov_{n}": cov[n] for n in FEATURE_NAMES})
            rec["decode_ok"] = True
    except Exception as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    tmp = out_json + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, out_json)
    return (sd, clip_id, rec["decode_ok"], rec.get("error"))


def build_tasks(chosen, corpus, wavroot, out):
    log("indexing phone_segments.parquet ...")
    keep = {c["speaker_dir"] for c in chosen}
    ph = pd.read_parquet(corpus / "phone_segments.parquet",
                         columns=["clip_id", "speaker_dir", "phone", "start_s", "end_s"])
    ph = ph[ph["speaker_dir"].isin(keep)].sort_values(["speaker_dir", "clip_id", "start_s"])
    grouped = {}
    for (sd, clip), g in ph.groupby(["speaker_dir", "clip_id"], sort=False):
        grouped[(sd, clip)] = list(zip(g["phone"], g["start_s"], g["end_s"]))
    log(f"phones indexed for {len(grouped)} clips")

    tasks, skipped, no_wav, no_phones = [], 0, 0, 0
    for c in chosen:
        sd, sex, rel = c["speaker_dir"], c["sex"], c["rel_path"]
        for clip in c["clips"].split(","):
            if not clip:
                continue
            stem = clip[:-4]
            out_json = str(out / rel / f"{stem}.json")
            if os.path.exists(out_json):
                skipped += 1
                continue
            wav = str(wavroot / sd / f"{stem}.wav")
            if not os.path.exists(wav):
                no_wav += 1
                continue
            phrows = grouped.get((sd, clip))
            if not phrows:
                no_phones += 1
                continue
            tasks.append((sd, sex, clip, phrows, wav, out_json))
    log(f"tasks={len(tasks)} (already-done={skipped}, missing-wav={no_wav}, missing-phones={no_phones})")
    return tasks, skipped


def assemble_parquet(chosen, out: Path):
    log("assembling all_utterances.parquet from per-clip JSONs ...")
    rows = []
    for c in chosen:
        d = out / c["rel_path"]
        if not d.is_dir():
            continue
        for jf in d.glob("*.json"):
            try:
                rows.append(json.load(open(jf)))
            except Exception:
                pass
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(out / "all_utterances.parquet", index=False)
        ok = int(df.get("decode_ok", pd.Series(dtype=bool)).sum())
        log(f"all_utterances.parquet: {ok}/{len(df)} decode_ok -> {out/'all_utterances.parquet'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEF_CORPUS))
    ap.add_argument("--wavroot", default=str(DEF_WAVROOT))
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--manifest", default=str(DEF_MANIFEST))
    ap.add_argument("--n-per-sex", type=int, default=4000)
    ap.add_argument("--use-manifest", default="",
                    help="use a FROZEN feats manifest (e.g. manifest/feats_manifest_8000.tsv) "
                         "verbatim for the exact study speakers+sex, instead of re-selecting")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="cap tasks (smoke test)")
    ap.add_argument("--claudemd", default=str(DEF_CLAUDEMD),
                    help="CLAUDE.md whose LIVE PROGRESS line is rewritten every 1000 clips")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--no-assemble", action="store_true",
                    help="skip parquet assembly (for array tasks; assemble once at the end "
                         "with --assemble-only over the full manifest)")
    args = ap.parse_args()

    # keep per-worker BLAS/torch threads at 1 so `jobs` subprocesses don't oversubscribe 8 cores
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    import multiprocessing as mp
    corpus, wavroot, out = Path(args.corpus), Path(args.wavroot), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.use_manifest:
        chosen, n_f_pool, n_m_pool = load_chosen_manifest(Path(args.use_manifest))
        log(f"using frozen manifest {args.use_manifest} (no re-selection)")
    else:
        chosen, n_f_pool, n_m_pool = select(corpus, args.n_per_sex)
        write_manifest(chosen, Path(args.manifest))
    nF = sum(c["sex"] == "F" for c in chosen)
    nM = sum(c["sex"] == "M" for c in chosen)
    srcF = sorted({c["gender_source"] for c in chosen if c["sex"] == "F"})
    src_note = (f"frozen manifest {args.use_manifest}" if args.use_manifest
                else f"manifest -> {args.manifest}")
    log(f"selected {nF} female (pool {n_f_pool}) + {nM} male (pool {n_m_pool}) "
        f"= {len(chosen)} speakers; {src_note}")
    log(f"  female composition: " + ", ".join(
        f"{s}={sum(c['sex']=='F' and c['gender_source']==s for c in chosen)}" for s in ("self_reported", "estimated")))
    log(f"  male composition:   " + ", ".join(
        f"{s}={sum(c['sex']=='M' and c['gender_source']==s for c in chosen)}" for s in ("self_reported", "estimated")))

    if args.assemble_only:
        assemble_parquet(chosen, out)
        return

    tasks, n_skipped = build_tasks(chosen, corpus, wavroot, out)
    if args.limit:
        tasks = tasks[:args.limit]
    total = len(tasks)
    total_all = n_skipped + total          # cumulative target across restarts
    if not total:
        log("nothing to do (all clips already extracted)"
            + ("" if args.no_assemble else "; assembling parquet."))
        update_claudemd(args.claudemd, total_all, total_all, n_skipped, 0.0, 0.0)
        if not args.no_assemble:
            assemble_parquet(chosen, out)
        return

    log(f"extracting {total} clips with jobs={args.jobs} (DeepFormants+DeepFry per clip) "
        f"[{n_skipped} already done, cumulative target {total_all}] ...")
    t0 = time.time()
    done = ok = 0
    last_kmark = n_skipped // 1000         # CLAUDE.md is rewritten when cumulative crosses each 1000
    prog = out / "_progress.json"
    with mp.Pool(args.jobs, maxtasksperchild=200) as pool:
        for sd, clip, dok, err in pool.imap_unordered(_extract_one, tasks, chunksize=1):
            done += 1
            ok += int(dok)
            cum = n_skipped + done
            if done % 50 == 0 or done == total:
                rate = done / max(1e-9, time.time() - t0)
                eta_h = (total - done) / max(1e-9, rate) / 3600
                json.dump({"done": done, "total": total,
                           "cumulative_done": cum, "total_all": total_all, "ok": ok,
                           "rate_per_s": round(rate, 3), "eta_hours": round(eta_h, 2),
                           "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, open(prog, "w"))
                if done % 200 == 0 or done == total:
                    log(f"  {cum}/{total_all} (ok={ok}, {rate:.2f}/s, ETA {eta_h:.1f} h)"
                        + (f"  last_err={err}" if err else ""))
            if cum // 1000 > last_kmark:    # crossed a 1000-clip boundary -> checkpoint CLAUDE.md
                last_kmark = cum // 1000
                rate = done / max(1e-9, time.time() - t0)
                eta_h = (total - done) / max(1e-9, rate) / 3600
                update_claudemd(args.claudemd, cum, total_all, n_skipped + ok, rate, eta_h)

    update_claudemd(args.claudemd, n_skipped + done, total_all, n_skipped + ok,
                    done / max(1e-9, time.time() - t0),
                    (total - done) / max(1e-9, done / max(1e-9, time.time() - t0)) / 3600)
    log(f"extraction pass done in {(time.time()-t0)/3600:.2f} h ({ok}/{done} decode_ok)")
    if not args.no_assemble:
        assemble_parquet(chosen, out)
    log("=== extract_feats_8k done ===")


if __name__ == "__main__":
    main()
