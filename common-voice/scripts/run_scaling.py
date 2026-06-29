#!/usr/bin/env python
"""Speaker-count scaling analysis for the Common Voice 8k study, parametrized for PSC.

Runs the committed scaling sweep (src.scaling) over a balanced grid of TOTAL speakers
N = 1000, 2000, ..., 8000 (i.e. 500..4000 per sex) and emits four analyses:
  - pooled        : balanced mixed-gender subsets (the headline scaling curve)
  - within_sex    : sex-controlled (per-gender measured, weight-combined)
  - female_only   : F speakers only, at 500..4000
  - male_only     : M speakers only, at 500..4000
Each is PR (effective dim) / summed-MI (upper bound) / Fano (lower bound) vs N, with the
log2 N ceiling. Tables + a combined figure + a slopes summary are written under --out.

Run (env voice-is-unique):
  python run_scaling.py --parquet <all_utterances.parquet> --out <dir> \
         [--grid 500,1000,1500,2000,2500,3000,3500,4000] [--clips 10]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

CV_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CV_PKG))
from src import common, scaling as SC


def slopes(d):
    x = np.log2(d["S"].to_numpy())
    return {c: (float(np.polyfit(x, d[c].to_numpy(), 1)[0]) if len(x) > 1 else float("nan"))
            for c in ("PR", "fano", "summed_mi")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CV_PKG / "CONFIG" / "common_voice.json"))
    ap.add_argument("--parquet", default="", help="features parquet (overrides config input_parquet)")
    ap.add_argument("--grid", default="500,1000,1500,2000,2500,3000,3500,4000",
                    help="per-sex sizes; pooled/within use N=2x these (1000..8000 total)")
    ap.add_argument("--clips", type=int, default=0, help="clips/speaker (0 = config default)")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.parquet:
        cfg["input_parquet"] = args.parquet
    grid = [int(x) for x in args.grid.split(",") if x]
    m = args.clips or cfg["scaling"]["fixed_clips_per_speaker"]
    out = Path(args.out); (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "reports" / "figs").mkdir(parents=True, exist_ok=True)

    df = common.load_table(cfg)
    sk, xk = cfg["speaker_key"], cfg["sex_key"]
    print(f"[scaling] {df[sk].nunique()} speakers, grid/sex={grid}, m={m}", flush=True)

    runs = {
        "pooled":      SC.sweep(df, cfg, grid, m, "pooled"),
        "within_sex":  SC.sweep(df, cfg, grid, m, "within_sex"),
        "female_only": SC.sweep(df[df[xk] == "F"], cfg, grid, m, "pooled"),
        "male_only":   SC.sweep(df[df[xk] == "M"], cfg, grid, m, "pooled"),
    }
    for name, d in runs.items():
        d["analysis"] = name
        d.to_csv(out / "tables" / f"scaling_{name}.csv", index=False)
        s = slopes(d)
        print(f"[scaling] {name:12s} slopes vs log2 N: "
              f"PR={s['PR']:.2f} Fano={s['fano']:.2f} summedMI={s['summed_mi']:.2f}", flush=True)
    pd.concat(runs.values(), ignore_index=True).to_csv(out / "tables" / "scaling_all.csv", index=False)

    # combined figure: 4 panels
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
    for ax, name in zip(axes, runs):
        d = runs[name]; N = d["S"].to_numpy()
        ax.plot(N, d["ceiling"], "k--", label="ceiling log2 N")
        ax.plot(N, d["fano"], "-o", label="Fano (lower)")
        ax.plot(N, d["summed_mi"], "-s", label="summed-MI (upper)")
        ax.plot(N, d["PR"], "-^", label="PR (d_eff)")
        ax.set_xscale("log", base=2); ax.set_xlabel("speakers N"); ax.set_title(name); ax.legend(fontsize=7)
    fig.suptitle("Common Voice speaker-count scaling (balanced 1000..8000 + gender-specific)")
    fig.tight_layout(); fig.savefig(out / "reports" / "figs" / "scaling_all.png", dpi=120)
    print(f"[scaling] DONE -> {out}/tables/scaling_*.csv, reports/figs/scaling_all.png", flush=True)


if __name__ == "__main__":
    main()
