#!/bin/bash
# Concatenate every chunk's vot_pilot.tsv into one full table, then merge VOT into the
# 40-feature parquet's VOT column (#39). Runs after BOTH the VOT array and the feats
# assemble are done.
#SBATCH -A cis250145p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 0:30:00
#SBATCH -J cv_vot_merge
#SBATCH -o logs/vot_finalize_%j.out
set -euo pipefail
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
O=/ocean/projects/cis250019p/ramakrib
conda config --add envs_dirs "$O/vu/envs" 2>/dev/null || true
FEATS=$O/vu/data/cv_feats
VOT=$O/vu/data/cv_vot
REPO=$O/vu/repo/voice-is-unique/common-voice

# 1. concat per-chunk per-stop tables (header from the first, bodies from all)
mapfile -t TSVS < <(ls "$VOT"/chunk_*/vot_pilot.tsv 2>/dev/null)
test ${#TSVS[@]} -gt 0 || { echo "no chunk vot_pilot.tsv found under $VOT"; exit 1; }
head -1 "${TSVS[0]}" > "$VOT/vot_pilot.tsv"
for f in "${TSVS[@]}"; do tail -n +2 "$f" >> "$VOT/vot_pilot.tsv"; done
echo "merged ${#TSVS[@]} chunk tables -> $VOT/vot_pilot.tsv ($(( $(wc -l < "$VOT/vot_pilot.tsv") - 1 )) stops)"

# 2. fold per-clip usable VOT into the parquet's VOT column (in place)
cd "$REPO/scripts"
conda run -n voice-is-unique python -u merge_vot.py \
    --vot-table "$VOT/vot_pilot.tsv" --parquet "$FEATS/all_utterances.parquet"
echo "=== VOT merged into $FEATS/all_utterances.parquet ==="
