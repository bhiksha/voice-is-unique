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


# canonical panel/row order; combine uses whichever are actually present.
# CV uses 'pooled' (gender-balanced); TIMIT uses 'all' (entire, unbalanced).
ANALYSES = ["all", "pooled", "within_sex", "female_only", "male_only"]


def _df_for(df, cfg, name):
    """The speaker pool a given analysis draws from."""
    xk = cfg["sex_key"]
    if name == "female_only":
        return df[df[xk] == "F"]
    if name == "male_only":
        return df[df[xk] == "M"]
    return df  # all, pooled, within_sex


def _all_subset(df, cfg, n, m):
    """'Entire data' subset: first n of ALL speakers (sex-agnostic, seeded), m clips each.
    Unbalanced — keeps the corpus's natural sex ratio. Nested across n (same seeded order)."""
    sk = cfg["speaker_key"]
    spk = df.drop_duplicates(sk)[sk].to_numpy()
    rng = np.random.default_rng(cfg["scaling"]["nested_order_seed"])
    keep = spk[rng.permutation(len(spk))][:n]
    sub = df[df[sk].isin(keep)].copy()
    if "clip_id" in sub.columns:
        sub = sub.sort_values("clip_id")
    return sub.groupby(sk, group_keys=False).head(m).reset_index(drop=True)


def _figure(runs, out, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = [a for a in ANALYSES if a in runs]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.3), squeeze=False)
    for ax, name in zip(axes[0], names):
        d = runs[name].sort_values("S"); N = d["S"].to_numpy()
        ax.plot(N, d["ceiling"], "k--", label="ceiling log2 N")
        ax.plot(N, d["fano"], "-o", label="Fano (lower)")
        ax.plot(N, d["summed_mi"], "-s", label="summed-MI (upper)")
        ax.plot(N, d["PR"], "-^", label="PR (d_eff)")
        ax.set_xscale("log", base=2); ax.set_xlabel("speakers N"); ax.set_title(name); ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out / "reports" / "figs" / "scaling_all.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CV_PKG / "CONFIG" / "common_voice.json"))
    ap.add_argument("--parquet", default="", help="features parquet (overrides config input_parquet)")
    ap.add_argument("--grid", default="500,1000,1500,2000,2500,3000,3500,4000",
                    help="per-sex sizes; pooled/within use N=2x these (1000..8000 total)")
    ap.add_argument("--clips", type=int, default=0, help="clips/speaker (0 = config default)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--only", default="",
                    choices=["", "all", "pooled", "within_sex", "female_only", "male_only"],
                    help="compute ONE analysis at ONE size (--n); write tables/pt_<only>_<n>.csv "
                         "(for a SLURM array — one job per analysis x size point). "
                         "'all' = entire (unbalanced) set; --n is the TOTAL speaker count there")
    ap.add_argument("--n", type=int, default=0,
                    help="per-sex size for --only (TOTAL speaker count for --only all)")
    ap.add_argument("--combine", action="store_true",
                    help="stitch all tables/pt_*.csv into scaling_*.csv + figure + "
                         "reports/scaling_report.txt")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.parquet:
        cfg["input_parquet"] = args.parquet
    m = args.clips or cfg["scaling"]["fixed_clips_per_speaker"]
    out = Path(args.out); (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "reports" / "figs").mkdir(parents=True, exist_ok=True)

    # ── combine mode: stitch the array's per-point CSVs into final tables/figure/report ──
    if args.combine:
        import glob
        pts = sorted(glob.glob(str(out / "tables" / "pt_*.csv")))
        if not pts:
            sys.exit("[scaling] --combine: no tables/pt_*.csv found")
        allrows = pd.concat([pd.read_csv(p) for p in pts], ignore_index=True)
        corpus = cfg.get("corpus", "")
        runs, lines = {}, [f"===== {corpus} speaker-count scaling — final report =====",
                           f"points={len(allrows)}  analyses={sorted(allrows['analysis'].unique())}", ""]
        for name in ANALYSES:
            d = allrows[allrows["analysis"] == name].sort_values("S").reset_index(drop=True)
            if d.empty:
                continue
            runs[name] = d
            d.to_csv(out / "tables" / f"scaling_{name}.csv", index=False)
            s = slopes(d)
            lines.append(f"[{name}]  slopes vs log2 N:  PR={s['PR']:.3f}  Fano={s['fano']:.3f}  "
                         f"summedMI={s['summed_mi']:.3f}")
            lines.append("       N    ceiling       PR      Fano   summedMI")
            for _, r in d.iterrows():
                lines.append(f"   {int(r['S']):5d}   {r['ceiling']:7.3f}  {r['PR']:7.3f}  "
                             f"{r['fano']:7.3f}  {r['summed_mi']:8.3f}")
            lines.append("")
        allrows.sort_values(["analysis", "S"]).to_csv(out / "tables" / "scaling_all.csv", index=False)
        if runs:
            _figure(runs, out, f"{corpus} speaker-count scaling (PR / Fano / summed-MI vs N)")
        (out / "reports" / "scaling_report.txt").write_text("\n".join(lines) + "\n")
        for ln in lines:
            print(ln, flush=True)
        print(f"[scaling] COMBINE DONE -> {out}/tables/scaling_*.csv, "
              f"reports/scaling_report.txt, reports/figs/scaling_all.png", flush=True)
        return

    df = common.load_table(cfg)
    sk = cfg["speaker_key"]

    # ── single-point mode: one (analysis, n) point for a SLURM array task ──
    if args.only:
        if args.n <= 0:
            sys.exit("--only requires --n <per-sex size>")
        ptf = out / "tables" / f"pt_{args.only}_{args.n}.csv"
        if ptf.exists():
            print(f"[scaling] {ptf.name} exists; skip", flush=True)
            return
        if args.only == "all":
            sub = _all_subset(df, cfg, args.n, m)        # entire, unbalanced (natural sex ratio)
            r = SC.measure(sub, cfg)
            cond = "all"
        else:
            base = _df_for(df, cfg, args.only)
            order = SC.nested_order(base, cfg)           # seed-deterministic -> matches the in-process sweep
            sub = SC.subset(base, cfg, args.n, order, m)
            r = SC.measure_within_sex(sub, cfg) if args.only == "within_sex" else SC.measure(sub, cfg)
            cond = "within_sex" if args.only == "within_sex" else "pooled"
        r = dict(r)
        r["n_per_sex"] = args.n
        r["condition"] = cond
        r["analysis"] = args.only
        pd.DataFrame([r]).to_csv(ptf, index=False)
        print(f"[scaling] point {args.only} n/sex={args.n} S={int(r['S'])} "
              f"PR={r['PR']:.3f} Fano={r['fano']:.3f} summedMI={r['summed_mi']:.3f} -> {ptf.name}", flush=True)
        return

    # ── default: full in-process sweep (unchanged behavior) ──
    grid = [int(x) for x in args.grid.split(",") if x]
    xk = cfg["sex_key"]
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
    _figure(runs, out, f"{cfg.get('corpus','')} speaker-count scaling (PR / Fano / summed-MI vs N)")
    print(f"[scaling] DONE -> {out}/tables/scaling_*.csv, reports/figs/scaling_all.png", flush=True)


if __name__ == "__main__":
    main()
