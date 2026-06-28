#!/usr/bin/env python
"""Verify that speaker-level embeddings can be formed by AVERAGING the per-clip
features for every speaker, using the SAME mechanism as the TIMIT / pilot analysis
(common.transformed_matrix + common.speaker_means): per feature, a speaker's
embedding value is the mean over that speaker's NON-missing (finite) clips, and is
NaN ("fails") only when the speaker has no measured value for that feature.

What it does:
  1. load all_utterances.parquet + CONFIG (feature list, transforms, VFP=VFI).
  2. build the utterance matrix exactly as the analysis does: per-feature transform
     (log for NAQ/alpha_ratio/LHR/SPI; linear otherwise) + corpus-wide z-score, NaNs
     preserved; VFI carried raw (its hurdle is downstream).
  3. average to a speaker x feature embedding via common.speaker_means.
  4. report, per feature, how many of the S speakers FAIL (NaN embedding); and, per
     speaker, which features it is missing. Writes the embedding + a failures table.

Outputs (under feats_dir):
  speaker_embeddings.parquet   S speakers x (40 features) averaged embedding (NaN=fail)
  embedding_failures.tsv       one row per (speaker, feature) that failed, with how
                               many of the speaker's clips were finite for it
  embedding_verify_report.txt  human-readable summary

Run (env voice-is-unique):
  python verify_speaker_embeddings.py [--config ../CONFIG/common_voice.json]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CV_PKG = HERE.parent
sys.path.insert(0, str(CV_PKG))
from src import common  # the shared TIMIT/pilot analysis core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CV_PKG / "CONFIG" / "common_voice.json"))
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    spk_key = cfg["speaker_key"]
    sex_key = cfg.get("sex_key", "sex")
    vfp = cfg["vfp"]["name"]
    feats_dir = Path(cfg["feats_dir"]).expanduser()

    df = common.load_table(cfg)
    print(f"loaded {len(df)} utterances, {df[spk_key].nunique()} speakers", flush=True)

    speakers = common.sorted_speakers(df, cfg)
    spk_ids = df[spk_key].to_numpy()

    # --- same mechanism: transform + z-score (NaN-preserved) for non-VFP features ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # all-NaN features (e.g. VOT) -> NaN column
        X, names = common.transformed_matrix(df, cfg)
        emb = {f: common.speaker_means(X[:, j], spk_ids, speakers) for j, f in enumerate(names)}
        # VFI carried raw (hurdle handled downstream); average the same NaN-aware way
        emb[vfp] = common.speaker_means(df[vfp].to_numpy(float), spk_ids, speakers)

    # restore canonical feature order
    order = [f for f in cfg["feature_names"]]
    E = pd.DataFrame({f: emb[f] for f in order}, index=speakers)
    E.index.name = spk_key

    # per-speaker sex + clip count (for the report)
    sex = df.drop_duplicates(spk_key).set_index(spk_key)[sex_key].reindex(speakers)
    nclip = df.groupby(spk_key).size().reindex(speakers)

    # --- failure analysis ---
    miss = E.isna()
    per_feature_fail = miss.sum(axis=0).sort_values(ascending=False)
    global_missing = [f for f in order if int(per_feature_fail[f]) == len(speakers)]
    usable = [f for f in order if f not in global_missing]

    # finite-clip count per (speaker, feature) so failures show 0/N support
    finite_counts = {}
    for f in order:
        v = df[f].to_numpy(float)
        fc = pd.Series(np.isfinite(v).astype(int), index=df[spk_key].values)
        finite_counts[f] = fc.groupby(level=0).sum().reindex(speakers).fillna(0).astype(int)

    rows = []
    for f in order:
        for s in speakers[miss[f].to_numpy()]:
            rows.append((s, sex.get(s, ""), int(nclip.get(s, 0)), f,
                         int(finite_counts[f].get(s, 0)),
                         "global" if f in global_missing else "speaker"))
    fails = pd.DataFrame(rows, columns=[spk_key, "sex", "n_clips", "feature",
                                        "finite_clips", "scope"])

    # speakers that fail on a USABLE feature (i.e. beyond the global all-NaN ones)
    usable_miss = miss[usable]
    bad_speakers = speakers[usable_miss.any(axis=1).to_numpy()]

    feats_dir.mkdir(parents=True, exist_ok=True)
    E.to_parquet(feats_dir / "speaker_embeddings.parquet")
    fails.to_csv(feats_dir / "embedding_failures.tsv", sep="\t", index=False)

    # --- report ---
    lines = []
    lines.append("===== speaker-embedding-by-averaging verification =====")
    lines.append(f"utterances={len(df)}  speakers={len(speakers)}  features={len(order)}")
    lines.append(f"embedding matrix: {E.shape[0]} speakers x {E.shape[1]} features "
                 f"-> {feats_dir/'speaker_embeddings.parquet'}")
    lines.append("")
    lines.append("GLOBALLY UNAVERAGEABLE features (NaN for ALL speakers):")
    lines.append("  " + (", ".join(global_missing) if global_missing else "(none)"))
    lines.append("")
    lines.append(f"Usable features: {len(usable)} of {len(order)}")
    lines.append(f"Speakers with a COMPLETE embedding over usable features: "
                 f"{len(speakers) - len(bad_speakers)} / {len(speakers)}")
    lines.append(f"Speakers FAILING >=1 usable feature: {len(bad_speakers)}")
    lines.append("")
    lines.append("Per-feature failure counts (speakers with NaN embedding):")
    for f in order:
        c = int(per_feature_fail[f])
        tag = "  [GLOBAL]" if f in global_missing else ("" if c == 0 else "  <-- partial")
        lines.append(f"  {f:20s} {c:6d}/{len(speakers)}{tag}")
    lines.append("")
    if len(bad_speakers):
        lines.append("Speakers failing a usable feature (first 30):")
        sub = fails[(fails.scope == "speaker")].groupby(spk_key)["feature"].apply(
            lambda s: ",".join(sorted(s)))
        for s in list(bad_speakers)[:30]:
            lines.append(f"  {s:12s} sex={sex.get(s,'?')} missing={sub.get(s,'')}")
    else:
        lines.append("Every speaker has a complete averaged embedding over all usable "
                     "features. (Only the GLOBAL features above are unaverageable.)")
    report = "\n".join(lines)
    (feats_dir / "embedding_verify_report.txt").write_text(report + "\n")
    print(report, flush=True)
    print(f"\nfailures table -> {feats_dir/'embedding_failures.tsv'} ({len(fails)} rows)", flush=True)


if __name__ == "__main__":
    main()
