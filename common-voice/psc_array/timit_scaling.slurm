#!/bin/bash
# TIMIT speaker-count scaling, run through the SAME code as the CV analysis
# (run_scaling.py + common-voice/src/) for cross-corpus parity. Ground-truth gender.
# Four analyses with per-analysis grids (TIMIT = 630 spk, 438 M / 192 F):
#   all         (entire, UNBALANCED set)         N = 100..630
#   within_sex  (sex-controlled, weight-combined) per-sex 40..192
#   female_only                                   40..192
#   male_only                                     100..438
# TIMIT is small (<=630-way Fano), so one looping job, not an array.
#SBATCH -A cis250145p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 4:00:00
#SBATCH -J timit_scal
#SBATCH -o logs/timit_scal_%j.out
set -euo pipefail
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
O=/ocean/projects/cis250019p/ramakrib
conda config --add envs_dirs "$O/vu/envs" 2>/dev/null || true
REPO=$O/vu/repo/voice-is-unique/common-voice
TPARQUET=$O/vu/data/timit-feats/all_utterances.parquet
TOUT=$O/vu/data/timit-analysis
CFG=CONFIG/timit.json
cd "$REPO"
run() { conda run -n voice-is-unique python -u scripts/run_scaling.py \
          --config "$CFG" --parquet "$TPARQUET" --out "$TOUT" --only "$1" --n "$2"; }
for n in 100 200 300 400 500 630; do run all "$n"; done
for n in 40 80 120 160 192;       do run within_sex "$n"; run female_only "$n"; done
for n in 100 200 300 400 438;     do run male_only "$n"; done
conda run -n voice-is-unique python -u scripts/run_scaling.py --config "$CFG" --out "$TOUT" --combine
echo "=== TIMIT scaling report ==="
cat "$TOUT/reports/scaling_report.txt"
