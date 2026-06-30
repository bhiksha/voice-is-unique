#!/bin/bash
# 40-feature extraction (39 feats; VOT done separately) as a SLURM array.
# Each task = one manifest chunk -> disjoint speakers -> disjoint per-clip JSONs under $FEATS.
# Resumable (skips any clip whose JSON exists). --no-assemble: parquet built once afterwards.
#SBATCH -A cis250145p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 2:00:00
#SBATCH -J cv_feats
#SBATCH -o logs/feats_%A_%a.out
#SBATCH --array=0-79
set -euo pipefail
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
O=/ocean/projects/cis250019p/ramakrib
conda config --add envs_dirs "$O/vu/envs" 2>/dev/null || true
DATA=$O/vu/data/cv_study
FEATS=$O/vu/data/cv_feats
REPO=$O/vu/repo/voice-is-unique/common-voice
CH=$(printf "%s/vu/chunks/chunk_%03d.tsv" "$O" "$SLURM_ARRAY_TASK_ID")
cd "$REPO/scripts"
echo "[task $SLURM_ARRAY_TASK_ID] chunk=$CH  $(date)"
conda run -n voice-is-unique python -u extract_feats_8k.py \
    --use-manifest "$CH" --no-assemble --claudemd /dev/null \
    --corpus "$DATA" --wavroot "$DATA/wavs" --out "$FEATS" --jobs 12
echo "[task $SLURM_ARRAY_TASK_ID] done  $(date)"
