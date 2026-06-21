"""Common Voice end-to-end driver.

    python -m src.run_all --config CONFIG/common_voice.json [--pilot]

Stages (each is a separate committed module so a reviewer reproduces from scratch):
  1. select   — src.download : choose 5000+5000 speakers, 100 clips each, manifest
  2. extract  — src.extract  : MFA-mask + 40-feature extraction → commonvoice-feats parquet
  3. analyze  — this module  : PR/MI/Fano scaling sweep (pooled + within-sex) → tables,
                               figures, report.

Stages 1–2 are gated on the Common Voice download (HF terms + login) and MFA, and
write only to ~/data (never the repo). This module runs the scaling sweep on the
already-extracted parquet and is the headline-figure producer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import common, scaling as SC


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="CONFIG/common_voice.json")
    ap.add_argument("--pilot", action="store_true", help="use the small pilot size grid")
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    parquet = Path(cfg["input_parquet"]).expanduser()
    if not parquet.exists():
        raise SystemExit(
            f"[run_all] features parquet not found: {parquet}\n"
            "Run the gated stages first: (1) python -m src.download --metadata-tsv ... ; "
            "(2) python -m src.extract --config CONFIG/common_voice.json")

    df = common.load_table(cfg)
    root = Path(args.out)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "figs").mkdir(parents=True, exist_ok=True)

    grid = cfg["scaling"]["pilot_grid_per_sex"] if args.pilot else cfg["scaling"]["full_grid_per_sex"]
    m = cfg["scaling"]["pilot_clips_per_speaker"] if args.pilot else cfg["scaling"]["fixed_clips_per_speaker"]
    tag = "pilot" if args.pilot else "full"

    print(f"[run_all] {tag} sweep grid (per sex) = {grid}, fixed clips/speaker = {m}", flush=True)
    pooled = SC.sweep(df, cfg, grid, m, condition="pooled")
    within = SC.sweep(df, cfg, grid, m, condition="within_sex")
    pooled.to_csv(root / "tables" / f"scaling_pooled_{tag}.csv", index=False)
    within.to_csv(root / "tables" / f"scaling_within_sex_{tag}.csv", index=False)
    SC.scaling_figure(pooled, within, root / "reports" / "figs" / f"scaling_{tag}.png")

    # slope of each curve vs log2 N (≈1 ⇒ pinned to the ceiling)
    def slope(d, col):
        x = np.log2(d["S"].to_numpy()); y = d[col].to_numpy()
        return float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else float("nan")
    print("[run_all] pooled slopes vs log2 N: PR=%.2f Fano=%.2f summedMI=%.2f (ceiling slope 1.0)"
          % (slope(pooled, "PR"), slope(pooled, "fano"), slope(pooled, "summed_mi")), flush=True)
    print(f"DONE ({tag}). tables/scaling_*_{tag}.csv, reports/figs/scaling_{tag}.png", flush=True)


if __name__ == "__main__":
    main()
