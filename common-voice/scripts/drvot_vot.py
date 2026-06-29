#!/usr/bin/env python
"""Estimate Voice Onset Time for voiceless stops (P/T/K) with Dr.VOT, using stop
windows derived from the MFA alignments (bypasses Dr.VOT's GTK-dependent Praat
voice-start detector — we already know the post-stop vowel onset).

For each P/T/K segment in phone_segments.parquet we cut a window from the clip's
16 kHz wav, set the "voice start" = the stop's end (= following-vowel onset) and write
Dr.VOT's voice_starts.txt ourselves, then run Dr.VOT's own feature extractor
(linux_VotFrontEnd2) + RNN model. Output: a per-stop VOT table + a quality report
(VOT distribution by phone, and yield vs the noise/bandwidth audit).

Pilot:  python drvot_vot.py --speakers 50 --out ~/data/commonvoice-feats/vot_pilot
"""
from __future__ import annotations
import argparse, csv, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf

HOME = Path.home()
# Dr.VOT is vendored in the repo (<repo>/extractor/third_party/drvot); env-overridable.
DRVOT = Path(os.environ.get("DRVOT_DIR",
            Path(__file__).resolve().parent.parent.parent / "extractor" / "third_party" / "drvot"))
DRVOT_PY = Path(os.environ.get("DRVOT_PY", HOME / "miniconda3/envs/drvot/bin/python"))
CORPUS = Path(os.environ.get("CV_CORPUS", HOME / "data/commonvoice"))
WAVROOT = Path(os.environ.get("CV_WAVROOT", HOME / "cv_align/corpus"))
MANIFEST = Path(__file__).resolve().parent.parent / "manifest" / "feats_manifest_8000.tsv"
AUDIT = HOME / "data/commonvoice-feats/noise_audit.parquet"
VOICELESS = {"P", "T", "K"}
SR = 16000
CUT_PRE, CUT_POST = 0.05, 0.30      # cut window: stop_start-PRE .. stop_end+POST
DRVOT_PRE, DRVOT_WIN = 0.05, 0.25   # Dr.VOT analysis window (model defaults)


def log(*a): print(*a, flush=True)
def natkey(sd): p, g, i = sd.split("_"); return (p, int(g), int(i))


def select(n):
    rows = list(csv.DictReader(open(MANIFEST), delimiter="\t"))
    pools = {"F": [], "M": []}
    for r in rows:
        pools[r["sex"]].append(r)
    out = []
    for sx in ("F", "M"):
        pools[sx].sort(key=lambda r: natkey(r["speaker_dir"]))
        out += pools[sx][: n // 2]
    return out


def cut_windows(chosen, raws):
    """Cut a window wav per voiceless stop, distributed ROUND-ROBIN across the shard
    dirs in `raws` (a list of Paths). Writes each shard's voice_starts.txt + files.txt.
    Returns the mapping DataFrame (with a 'shard' column)."""
    for r in raws:
        r.mkdir(parents=True, exist_ok=True)
    nshards = len(raws)
    keep = {c["speaker_dir"] for c in chosen}
    ph = pd.read_parquet(CORPUS / "phone_segments.parquet",
                         columns=["clip_id", "speaker_dir", "phone", "start_s", "end_s"])
    ph = ph[ph.speaker_dir.isin(keep)].sort_values(["speaker_dir", "clip_id", "start_s"])
    grouped = {k: list(zip(g.phone, g.start_s, g.end_s))
               for k, g in ph.groupby(["speaker_dir", "clip_id"], sort=False)}
    vs_lines = [[] for _ in raws]
    mapping = []
    n = 0
    for c in chosen:
        sd = c["speaker_dir"]
        for clip in c["clips"].split(","):
            segs = grouped.get((sd, clip))
            if not segs:
                continue
            wavp = WAVROOT / sd / f"{clip[:-4]}.wav"
            if not wavp.exists():
                continue
            x, sr = sf.read(wavp)
            if x.ndim > 1: x = x.mean(1)
            dur_clip = len(x) / sr
            for i, (lab, s, e) in enumerate(segs):
                if lab not in VOICELESS:
                    continue
                cut_a = max(0.0, s - CUT_PRE)
                cut_b = min(dur_clip, e + CUT_POST)
                seg = x[int(cut_a * sr):int(cut_b * sr)]
                if seg.size < int(0.06 * sr):
                    continue
                k = n % nshards
                n += 1
                name = f"{clip[:-4]}__s{i}_{lab}"
                wpath = raws[k] / f"{name}.wav"
                sf.write(wpath, seg.astype(np.float32), sr, subtype="PCM_16")
                # Anchor Dr.VOT's analysis window at the MFA stop ONSET so it spans the
                # whole closure->burst->voicing (window = [vstart-pre, vstart-pre+wsize]).
                # vstart-pre must land on stop_start_in_cut (= s-cut_a), so vstart = that + pre.
                vstart = round((s - cut_a) + DRVOT_PRE, 4)
                vdur = round(len(seg) / sr, 4) - 0.0001
                vs_lines[k].append(f"{wpath} : {vstart}  :{vdur}")
                mapping.append(dict(window=name, clip_id=clip, speaker_dir=sd, sex=c["sex"], phone=lab, shard=k))
    for k, r in enumerate(raws):
        (r / "voice_starts.txt").write_text("\n".join(vs_lines[k]) + "\n")
        (r / "files.txt").write_text("\n".join(l.split(" : ")[0].rsplit("/",1)[-1][:-4]
                                               for l in vs_lines[k]) + "\n")
    log(f"cut {len(mapping)} P/T/K windows from {len(chosen)} speakers into {nshards} shard(s)")
    return pd.DataFrame(mapping)


def run_drvot(raw: Path, work: Path):
    proc, out = work / "processed", work / "out"
    for d in (proc, out): d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env["PYTHONPATH"] = str(DRVOT)
    log("Dr.VOT: feature extraction (linux_VotFrontEnd2) ...")
    subprocess.run([str(DRVOT_PY), "-m", "process_data.feature_extractor",
                    str(raw), str(proc), "--window_size", str(DRVOT_WIN), "--pre", str(DRVOT_PRE),
                    "--prefix", "voiced", "--test",
                    "--features_file", "process_data/linux_VotFrontEnd2"],
                   cwd=str(DRVOT), env=env, check=True)
    log("Dr.VOT: predict ...")
    subprocess.run([str(DRVOT_PY), "predict.py", "--inference", str(proc),
                    "--out_dir", str(out), "--durations", str(raw / "voice_starts.txt")],
                   cwd=str(DRVOT), env=env, check=True)
    return out / "summary.csv"


_SHARD_IGNORE = shutil.ignore_patterns("linux_praat", "figures", ".git", "data",
                                       "__pycache__", "*.pyc")


def prep_shard_drvot(dst: Path):
    """Light copy of the vendored Dr.VOT (excludes the 35 MB unused GUI praat) so each
    shard has its own cwd for relative binary/model paths + isolated temp files."""
    if (dst / "predict.py").exists():
        return dst
    shutil.copytree(DRVOT, dst, ignore=_SHARD_IGNORE)
    os.chmod(dst / "process_data" / "linux_VotFrontEnd2", 0o755)
    return dst


def run_shard(k: int, shard_drvot: Path, raw: Path, proc: Path, out: Path, logf: Path):
    """Background process: feature-extract then predict for one shard. Resumable:
    skips if out/summary.csv already has all rows. Returns a Popen."""
    for d in (proc, out):
        d.mkdir(parents=True, exist_ok=True)
    nwin = sum(1 for _ in open(raw / "files.txt")) - (0 if (raw / "files.txt").read_text().endswith("\n") else 0)
    summ = out / "summary.csv"
    if summ.exists() and sum(1 for _ in open(summ)) - 1 >= max(1, nwin) * 0.99:
        log(f"  shard {k}: already complete ({summ}); skipping")
        return None
    fe = (f'cd {shard_drvot} && PYTHONPATH={shard_drvot} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
          f'{DRVOT_PY} -u -m process_data.feature_extractor {raw} {proc} '
          f'--window_size {DRVOT_WIN} --pre {DRVOT_PRE} --prefix voiced --test '
          f'--features_file process_data/linux_VotFrontEnd2')
    pr = (f'cd {shard_drvot} && PYTHONPATH={shard_drvot} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
          f'{DRVOT_PY} -u predict.py --inference {proc} --out_dir {out} '
          f'--durations {raw}/voice_starts.txt')
    cmd = f"({fe}) && ({pr})"
    return subprocess.Popen(["bash", "-c", cmd], stdout=open(logf, "w"), stderr=subprocess.STDOUT)


def run_sharded(work: Path, nshards: int, max_parallel: int):
    """Run shards with at most `max_parallel` concurrent (each Dr.VOT predict holds
    ~the shard's features in RAM, so concurrency is memory-bound, not core-bound)."""
    import time as _t
    todo = list(range(nshards))
    running = {}   # k -> Popen
    done = 0
    while todo or running:
        while todo and len(running) < max_parallel:
            k = todo.pop(0)
            sd = work / f"shard{k}"
            drv = prep_shard_drvot(sd / "drvot")
            p = run_shard(k, drv, sd / "all_files", sd / "processed", sd / "out", sd / "shard.log")
            if p is None:        # already complete -> skip
                done += 1
            else:
                running[k] = p
        for k, p in list(running.items()):
            if p.poll() is not None:
                done += 1
                log(f"  shard {k} finished rc={p.returncode} ({done}/{nshards} done, "
                    f"{len(running)-1} running, {len(todo)} queued)")
                del running[k]
        _t.sleep(5)
    return [work / f"shard{k}" / "out" / "summary.csv" for k in range(nshards)]


def merge_summaries(summaries):
    frames = []
    for s in summaries:
        if Path(s).exists():
            frames.append(pd.read_csv(s))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def report(summary, mapping: pd.DataFrame, work: Path):
    s = summary.copy() if isinstance(summary, pd.DataFrame) else pd.read_csv(summary)
    s.columns = [c.strip() for c in s.columns]
    s = s.rename(columns={s.columns[0]: "filename", s.columns[1]: "type", s.columns[2]: "vot_ms"})
    s["filename"] = s["filename"].astype(str).str.strip()
    s["vot_ms"] = pd.to_numeric(s["vot_ms"], errors="coerce")
    # Dr.VOT preserves our full unique window name (e.g. <clip>__s17_P) as the summary
    # filename stem -> match on that (the full name is unique per stop).
    s["window"] = s["filename"].map(lambda f: Path(str(f)).stem.strip())
    df = mapping.merge(s[["window", "type", "vot_ms"]].drop_duplicates("window"),
                       on="window", how="left")
    if AUDIT.exists():
        au = pd.read_parquet(AUDIT)[["clip_id", "snr_db", "bandwidth_hz"]]
        df = df.merge(au, on="clip_id", how="left")
    df.to_csv(work / "vot_pilot.tsv", sep="\t", index=False)

    got = df.vot_ms.notna()
    lines = ["===== Dr.VOT pilot (voiceless P/T/K, MFA windows) =====",
             f"windows: {len(df)} | VOT predicted: {int(got.sum())} ({100*got.mean():.1f}%)"]
    g = df[got]
    lines.append("")
    lines.append("VOT (ms) by stop:")
    for p in ["P", "T", "K"]:
        v = g[g.phone == p].vot_ms
        if len(v):
            lines.append(f"  {p}: n={len(v):5d} median={v.median():5.1f} IQR=[{v.quantile(.25):.0f},"
                         f"{v.quantile(.75):.0f}] mean={v.mean():5.1f}")
    if "type" in g:
        g = g.assign(type=g.type.astype(str).str.strip())
        lines.append("  type: " + ", ".join(f"{k}={v}" for k, v in g.type.value_counts().items()))
        neg = (g.type == "NEG_VOT").mean()
        lines.append(f"  NEG-tagged on voiceless (artifact): {100*neg:.1f}%")
    lines.append(f"  implausible (<3 or >150 ms): {int(((g.vot_ms<3)|(g.vot_ms>150)).sum())}")
    # 'usable' = positive VOT in a plausible range -> the scientifically usable measure
    use = g[(g.get('type','POS_VOT') == 'POS_VOT') & (g.vot_ms.between(3, 150))] if 'type' in g else g[g.vot_ms.between(3,150)]
    lines.append("")
    lines.append(f"USABLE (POS, 3-150 ms): {len(use)}/{len(df)} ({100*len(use)/max(1,len(df)):.1f}%)")
    for p in ["P", "T", "K"]:
        v = use[use.phone == p].vot_ms
        if len(v):
            lines.append(f"  {p}: n={len(v):5d} median={v.median():5.1f} IQR=[{v.quantile(.25):.0f},{v.quantile(.75):.0f}]")
    if "snr_db" in g:
        lines.append("")
        lines.append("yield / VOT vs quality:")
        for lab, m in [("SNR>=10dB", g.snr_db >= 10), ("SNR<10dB", g.snr_db < 10),
                       ("BW>=7kHz", g.bandwidth_hz >= 7000), ("BW<7kHz", g.bandwidth_hz < 7000)]:
            vv = g[m].vot_ms
            lines.append(f"  {lab:10s} n={int(m.sum()):5d} median_VOT={vv.median():5.1f}")
    rep = "\n".join(lines)
    (work / "vot_pilot_report.txt").write_text(rep + "\n")
    log("\n" + rep)
    log(f"\nper-stop table -> {work/'vot_pilot.tsv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, default=50, help="0 = all speakers in the manifest")
    ap.add_argument("--out", default=str(HOME / "data/commonvoice-feats/vot_pilot"))
    ap.add_argument("--shards", type=int, default=1, help="number of data shards (1 = single)")
    ap.add_argument("--max-parallel", type=int, default=0,
                    help="max concurrent shards (0 = all); memory-bound (~1.5 GB/predict)")
    ap.add_argument("--keep-windows", action="store_true")
    ap.add_argument("--mapping-only", action="store_true", help="(re)build mapping + report from existing shard summaries")
    ap.add_argument("--reuse-windows", action="store_true", help="skip cutting; reuse existing shard windows (e.g. after changing --max-parallel)")
    args = ap.parse_args()
    work = Path(args.out)
    chosen = select(args.speakers if args.speakers else 10 ** 9)
    log(f"selected {len(chosen)} speakers ({sum(c['sex']=='F' for c in chosen)} F / "
        f"{sum(c['sex']=='M' for c in chosen)} M)")

    if args.shards <= 1:
        raw = work / "all_files"
        if raw.exists() and not args.mapping_only:
            shutil.rmtree(raw)
        mapping = cut_windows(chosen, [raw]) if not args.mapping_only else _mapping_from(work, [raw])
        summary = run_drvot(raw, work) if not args.mapping_only else (work / "out" / "summary.csv")
        report(summary, mapping, work)
        if not args.keep_windows and not args.mapping_only:
            shutil.rmtree(raw, ignore_errors=True)
        return

    raws = [work / f"shard{k}" / "all_files" for k in range(args.shards)]
    if args.mapping_only:
        mapping = _mapping_from(work, raws)
    else:
        if args.reuse_windows:
            mapping = _mapping_from(work, raws)
        else:
            for r in raws:
                if r.exists(): shutil.rmtree(r)
            mapping = cut_windows(chosen, raws)
        run_sharded(work, args.shards, args.max_parallel or args.shards)
    merged = merge_summaries([work / f"shard{k}" / "out" / "summary.csv" for k in range(args.shards)])
    log(f"merged {len(merged)} VOT predictions across {args.shards} shards")
    report(merged, mapping, work)
    if not args.keep_windows and not args.mapping_only:
        for r in raws:
            shutil.rmtree(r, ignore_errors=True)


def _mapping_from(work: Path, raws):
    """Rebuild the stop->speaker/sex/phone mapping from the cut window wavs (for re-report)."""
    clip2 = {}
    for r in csv.DictReader(open(MANIFEST), delimiter="\t"):
        for c in r["clips"].split(","):
            clip2[c] = (r["speaker_dir"], r["sex"])
    rows = []
    for k, raw in enumerate(raws):
        for w in sorted(Path(raw).glob("*.wav")):
            nm = w.stem; clip = nm.split("__")[0] + ".mp3"; ph = nm.split("_")[-1]
            sd, sx = clip2.get(clip, ("?", "?"))
            rows.append(dict(window=nm, clip_id=clip, speaker_dir=sd, sex=sx, phone=ph, shard=k))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
