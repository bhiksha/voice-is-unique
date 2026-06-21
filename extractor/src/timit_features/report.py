"""Post-run QA / summary over all_utterances.parquet.

Reports, per feature: coverage (non-NaN fraction), basic distribution stats, and
the between-speaker / within-speaker variance ratio — an ICC-like index of how
speaker-distinctive each feature is (the load-bearing quantity for this study).
This is a read-only summary; the formal speaker-level statistics are a later stage.

Usage:
    python -m timit_features.report <feats_dir> [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from timit_features.config import FEATURE_NAMES


def coverage(df: pd.DataFrame) -> dict[str, float]:
    n = len(df)
    return {f: (float(np.isfinite(df[f]).mean()) if n else 0.0) for f in FEATURE_NAMES}


def distribution(df: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for f in FEATURE_NAMES:
        v = df[f].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        out[f] = (dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                       min=float(v.min()), max=float(v.max())) if v.size >= 2
                  else dict(n=int(v.size), mean=np.nan, sd=np.nan, min=np.nan, max=np.nan))
    return out


def speaker_variance_ratio(df: pd.DataFrame) -> dict[str, float]:
    """between-speaker variance / mean within-speaker variance, per feature.

    Higher ⇒ more speaker-distinctive. Uses speakers with >=2 finite utterances
    for the within-speaker term."""
    ratios = {}
    for f in FEATURE_NAMES:
        sub = df[["speaker_id", f]].dropna()
        if sub["speaker_id"].nunique() < 2:
            ratios[f] = float("nan")
            continue
        grp = sub.groupby("speaker_id")[f]
        spk_means = grp.mean()
        within = grp.var(ddof=1).dropna()           # per-speaker variance (>=2 utts)
        within_mean = float(within.mean()) if len(within) else float("nan")
        between = float(spk_means.var(ddof=1))
        ratios[f] = (between / within_mean) if within_mean and within_mean > 0 else float("nan")
    return ratios


def build_report(df: pd.DataFrame) -> dict:
    return {
        "n_utterances": int(len(df)),
        "n_speakers": int(df["speaker_id"].nunique()),
        "n_decode_failures": int((~df["decode_ok"].astype(bool)).sum()),
        "coverage": coverage(df),
        "distribution": distribution(df),
        "between_within_speaker_var_ratio": speaker_variance_ratio(df),
    }


def _fmt(rep: dict) -> str:
    lines = [
        f"utterances: {rep['n_utterances']}  speakers: {rep['n_speakers']}  "
        f"decode-failures: {rep['n_decode_failures']}", "",
        f"{'feature':16}{'cov%':>6}{'mean':>12}{'sd':>12}{'btw/within':>12}",
    ]
    cov, dist, rat = rep["coverage"], rep["distribution"], rep["between_within_speaker_var_ratio"]
    for f in FEATURE_NAMES:
        d = dist[f]
        mean = f"{d['mean']:.4g}" if np.isfinite(d['mean']) else "NaN"
        sd = f"{d['sd']:.4g}" if np.isfinite(d['sd']) else "NaN"
        r = f"{rat[f]:.3g}" if np.isfinite(rat[f]) else "NaN"
        lines.append(f"{f:16}{100*cov[f]:6.1f}{mean:>12}{sd:>12}{r:>12}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Summarize all_utterances.parquet")
    ap.add_argument("feats_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    df = pd.read_parquet(args.feats_dir / "all_utterances.parquet")
    rep = build_report(df)
    print(_fmt(rep))
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
