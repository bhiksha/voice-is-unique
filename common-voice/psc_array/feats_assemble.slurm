#!/bin/bash
# Build the single all_utterances.parquet from ALL per-clip JSONs once the array is done.
# Uses the FULL 8000-speaker manifest so the parquet is complete (--assemble-only globs JSONs).
#SBATCH -A cis250145p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 0:30:00
#SBATCH -J cv_feats_asm
#SBATCH -o logs/feats_assemble_%j.out
set -euo pipefail
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
O=/ocean/projects/cis250019p/ramakrib
conda config --add envs_dirs "$O/vu/envs" 2>/dev/null || true
DATA=$O/vu/data/cv_study
FEATS=$O/vu/data/cv_feats
REPO=$O/vu/repo/voice-is-unique/common-voice
cd "$REPO/scripts"
conda run -n voice-is-unique python -u extract_feats_8k.py \
    --use-manifest "$REPO/manifest/feats_manifest_8000.tsv" --assemble-only --claudemd /dev/null \
    --corpus "$DATA" --wavroot "$DATA/wavs" --out "$FEATS"
echo "rows in parquet:"
conda run -n voice-is-unique python -c "import pandas as pd; print(len(pd.read_parquet('$FEATS/all_utterances.parquet')))"
