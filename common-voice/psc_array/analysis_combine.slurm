#!/bin/bash
# Stitch the 32 per-point CSVs into the final per-analysis tables, the 4-panel figure,
# and reports/scaling_report.txt. Runs afterok of the analysis array.
#SBATCH -A cis250145p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 0:20:00
#SBATCH -J cv_scal_comb
#SBATCH -o logs/scal_combine_%j.out
set -euo pipefail
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
O=/ocean/projects/cis250019p/ramakrib
conda config --add envs_dirs "$O/vu/envs" 2>/dev/null || true
REPO=$O/vu/repo/voice-is-unique/common-voice
ANALYSIS=$O/vu/data/cv-analysis
cd "$REPO"
conda run -n voice-is-unique python -u scripts/run_scaling.py --combine --out "$ANALYSIS"
echo "=== final report ==="
cat "$ANALYSIS/reports/scaling_report.txt"
