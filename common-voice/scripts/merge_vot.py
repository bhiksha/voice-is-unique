#!/usr/bin/env python
"""Merge Dr.VOT measurements into the 40-feature parquet's VOT column.

The MFA-based extractor leaves VOT (#39) all-NaN on Common Voice (no closure/release
split). Dr.VOT (drvot_vot.py) measures it per voiceless stop; here we aggregate to a
per-CLIP VOT (mean of the clip's USABLE stops: positive VOT in 3-150 ms) and write it
into the parquet's `VOT` column (in SECONDS, matching the extractor's VOT units), so the
downstream analysis uses a real 40th feature instead of an empty one. Clips with no
usable VOT stay NaN (handled NaN-aware by speaker_means).

Run:  python merge_vot.py --vot-table <vot_pilot.tsv> --parquet <all_utterances.parquet>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vot-table", required=True, help="drvot_vot.py output (vot_pilot.tsv)")
    ap.add_argument("--parquet", required=True, help="all_utterances.parquet to update (in place)")
    ap.add_argument("--out", default="", help="output parquet (default: overwrite --parquet)")
    ap.add_argument("--lo", type=float, default=3.0)
    ap.add_argument("--hi", type=float, default=150.0)
    args = ap.parse_args()

    v = pd.read_csv(args.vot_table, sep="\t")
    v["vot_ms"] = pd.to_numeric(v["vot_ms"], errors="coerce")
    typ = v["type"].astype(str).str.strip() if "type" in v else pd.Series("POS_VOT", index=v.index)
    usable = v[(typ == "POS_VOT") & v["vot_ms"].between(args.lo, args.hi)]
    per_clip = usable.groupby("clip_id")["vot_ms"].mean() / 1000.0   # ms -> s
    print(f"[merge_vot] {len(v)} stops; {len(usable)} usable; "
          f"{per_clip.size} clips get a VOT", flush=True)

    df = pd.read_parquet(args.parquet)
    before = int(df["VOT"].notna().sum()) if "VOT" in df else 0
    df["VOT"] = df["clip_id"].map(per_clip)            # NaN where no usable stop
    after = int(df["VOT"].notna().sum())
    out = args.out or args.parquet
    df.to_parquet(out, index=False)
    print(f"[merge_vot] VOT non-null clips: {before} -> {after} / {len(df)} "
          f"({100*after/len(df):.1f}%); wrote {out}", flush=True)


if __name__ == "__main__":
    main()
