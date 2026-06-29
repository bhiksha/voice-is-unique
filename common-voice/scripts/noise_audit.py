#!/usr/bin/env python
"""Quick recording-quality audit over the study clips: alignment-based SNR, effective
spectral bandwidth (catches band-limited / codec / "phone" audio), and clipping.

SNR uses the MFA phones we already have: noise = sil/spn segments, signal = the rest.
SNR_dB = 10*log10((speech_power - noise_power)/noise_power). Bandwidth = highest freq
where the long-term spectrum stays within BW_DB of its peak. Outputs a per-clip table
+ a summary with how many clips fall below quality thresholds.

Run (env voice-is-unique):  python noise_audit.py [--jobs 8]
"""
from __future__ import annotations
import argparse, csv, json, os, time
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf

HOME = Path.home()
DEF_CORPUS = HOME / "data/commonvoice"
DEF_WAVROOT = HOME / "cv_align/corpus"
DEF_OUT = HOME / "data/commonvoice-feats"
NOISE_LABELS = {"sil", "spn", ""}
BW_DB = 30.0          # bandwidth = top freq within this many dB of spectral peak
SNR_LOW_DB = 10.0     # flag clips below this SNR
BW_LOW_HZ = 7000.0    # flag clips whose energy rolls off below this (band-limited)


def _clip_stats(task):
    clip_id, wav, phrows = task
    out = dict(clip_id=clip_id, snr_db=np.nan, bandwidth_hz=np.nan, clip_pct=np.nan, ok=False)
    try:
        x, sr = sf.read(wav, dtype="float64")
        if x.ndim > 1:
            x = x.mean(1)
        # alignment-based SNR
        npow = []; spow = []
        for ph, a, b in phrows:
            i0, i1 = int(round(a * sr)), int(round(b * sr))
            seg = x[i0:i1]
            if seg.size == 0:
                continue
            p = float(np.mean(seg ** 2))
            (npow if ph in NOISE_LABELS else spow).append((p, seg.size))
        def wmean(ps):
            if not ps:
                return np.nan
            w = np.array([n for _, n in ps]); v = np.array([p for p, _ in ps])
            return float((v * w).sum() / w.sum())
        nz, sg = wmean(npow), wmean(spow)
        if np.isfinite(nz) and np.isfinite(sg) and nz > 0:
            out["snr_db"] = 10 * np.log10(max(sg - nz, 1e-12) / nz)
        elif np.isfinite(sg):
            # no silence segment -> percentile-based noise floor from frame energies
            fr = x[: (len(x) // 320) * 320].reshape(-1, 320)
            fe = (fr ** 2).mean(1)
            nz = np.percentile(fe, 10); sg2 = np.percentile(fe, 90)
            if nz > 0:
                out["snr_db"] = 10 * np.log10(max(sg2 - nz, 1e-12) / nz)
        # effective bandwidth from long-term magnitude spectrum
        n = 1 << 15
        mag = np.abs(np.fft.rfft(x * np.hanning(len(x)) if len(x) < n else x[:n] * np.hanning(n), n=n))
        freqs = np.fft.rfftfreq(n, 1 / sr)
        sm = np.convolve(mag, np.ones(16) / 16, mode="same")
        peak = sm.max()
        if peak > 0:
            above = freqs[sm >= peak * 10 ** (-BW_DB / 20)]
            out["bandwidth_hz"] = float(above.max()) if above.size else 0.0
        out["clip_pct"] = float(np.mean(np.abs(x) > 0.999) * 100)
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEF_CORPUS))
    ap.add_argument("--wavroot", default=str(DEF_WAVROOT))
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--manifest", default=str(Path(__file__).resolve().parent.parent /
                                             "manifest" / "feats_manifest_8000.tsv"))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    import multiprocessing as mp
    corpus, wavroot, out = Path(args.corpus), Path(args.wavroot), Path(args.out)

    chosen = list(csv.DictReader(open(args.manifest), delimiter="\t"))
    keep = {c["speaker_dir"] for c in chosen}
    print(f"{len(chosen)} speakers; indexing phones ...", flush=True)
    ph = pd.read_parquet(corpus / "phone_segments.parquet",
                         columns=["clip_id", "speaker_dir", "phone", "start_s", "end_s"])
    ph = ph[ph["speaker_dir"].isin(keep)]
    grouped = {(sd, c): list(zip(g["phone"], g["start_s"], g["end_s"]))
               for (sd, c), g in ph.groupby(["speaker_dir", "clip_id"], sort=False)}

    tasks = []
    for c in chosen:
        sd = c["speaker_dir"]
        for clip in c["clips"].split(","):
            if not clip:
                continue
            wav = str(wavroot / sd / f"{clip[:-4]}.wav")
            pr = grouped.get((sd, clip))
            if pr and os.path.exists(wav):
                tasks.append((clip, wav, pr))
    print(f"auditing {len(tasks)} clips with jobs={args.jobs} ...", flush=True)

    t0 = time.time(); rows = []
    with mp.Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_clip_stats, tasks, chunksize=32), 1):
            rows.append(r)
            if i % 10000 == 0:
                print(f"  {i}/{len(tasks)} ({i/(time.time()-t0):.0f}/s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "noise_audit.parquet", index=False)

    g = df[df.ok]
    def pct(mask): return f"{int(mask.sum())} ({100*mask.mean():.1f}%)"
    lines = []
    lines.append("===== recording-quality audit =====")
    lines.append(f"clips audited: {len(g)} / {len(df)}")
    for col, lo in [("snr_db", None), ("bandwidth_hz", None), ("clip_pct", None)]:
        v = g[col].dropna()
        lines.append(f"{col:14s} p05={np.percentile(v,5):8.2f} p50={np.percentile(v,50):8.2f} "
                     f"p95={np.percentile(v,95):8.2f} mean={v.mean():8.2f}")
    lines.append("")
    lines.append(f"low SNR (<{SNR_LOW_DB} dB):       {pct(g.snr_db < SNR_LOW_DB)}")
    lines.append(f"band-limited (<{BW_LOW_HZ:.0f} Hz):  {pct(g.bandwidth_hz < BW_LOW_HZ)}")
    lines.append(f"clipping (>1% samples):    {pct(g.clip_pct > 1.0)}")
    report = "\n".join(lines)
    (out / "noise_audit_report.txt").write_text(report + "\n")
    print(report, flush=True)
    print(f"\nper-clip table -> {out/'noise_audit.parquet'}", flush=True)


if __name__ == "__main__":
    main()
