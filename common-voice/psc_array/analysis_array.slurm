#!/bin/bash
# PNAS scaling analysis as a 32-task SLURM array: one job per (analysis x size) point.
# 4 analyses (pooled, within_sex, female_only, male_only) x 8 per-sex sizes (500..4000).
# Each task writes tables/pt_<analysis>_<n>.csv; analysis_combine.slurm stitches them.
# Results are bit-identical to the single-process sweep (global z-score on the full parquet
# + seed-deterministic speaker order), but wall-clock collapses to the slowest single point.
#SBATCH -A cis250145p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 6:00:00
#SBATCH -J cv_scal
#SBATCH -o logs/scal_%A_%a.out
#SBATCH --array=0-31
set -euo pipefail
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
O=/ocean/projects/cis250019p/ramakrib
conda config --add envs_dirs "$O/vu/envs" 2>/dev/null || true
REPO=$O/vu/repo/voice-is-unique/common-voice
FEATS=$O/vu/data/cv_feats
ANALYSIS=$O/vu/data/cv-analysis
NAMES=(pooled within_sex female_only male_only)
GRID=(500 1000 1500 2000 2500 3000 3500 4000)
i=$SLURM_ARRAY_TASK_ID
NAME=${NAMES[$((i/8))]}
N=${GRID[$((i%8))]}
cd "$REPO"
echo "[task $i] analysis=$NAME n/sex=$N  $(date)"
conda run -n voice-is-unique python -u scripts/run_scaling.py \
    --parquet "$FEATS/all_utterances.parquet" \
    --only "$NAME" --n "$N" --out "$ANALYSIS"
echo "[task $i] done  $(date)"
