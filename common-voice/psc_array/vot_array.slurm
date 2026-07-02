#!/bin/bash
# VOT via Dr.VOT as a SLURM array. Each task = one manifest chunk (disjoint speakers),
# driven by CV_MANIFEST, writing its own self-contained outputs under $VOT/chunk_NNN/
# (windows + per-shard summaries + chunk vot_pilot.tsv). Concatenated afterwards.
# Within a task we sub-shard the chunk (--shards 6 --max-parallel 6) to use the task's
# cores; Dr.VOT predict is memory-bandwidth-bound so ~6 concurrent is the sweet spot.
#SBATCH -A cis250145p
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 2:00:00
#SBATCH -J cv_vot
#SBATCH -o logs/vot_%A_%a.out
#SBATCH --array=0-79
set -euo pipefail
source /opt/packages/anaconda3-2024.10-1/etc/profile.d/conda.sh
O=/ocean/projects/cis250019p/ramakrib
conda config --add envs_dirs "$O/vu/envs" 2>/dev/null || true
DATA=$O/vu/data/cv_study
VOT=$O/vu/data/cv_vot
REPO=$O/vu/repo/voice-is-unique/common-voice
CH=$(printf "%s/vu/chunks/chunk_%03d.tsv" "$O" "$SLURM_ARRAY_TASK_ID")
OUT=$(printf "%s/chunk_%03d" "$VOT" "$SLURM_ARRAY_TASK_ID")
cd "$REPO/scripts"
echo "[task $SLURM_ARRAY_TASK_ID] chunk=$CH out=$OUT  $(date)"
CV_MANIFEST="$CH" CV_CORPUS="$DATA" CV_WAVROOT="$DATA/wavs" \
DRVOT_PY="$O/vu/envs/drvot/bin/python" \
conda run -n voice-is-unique python -u drvot_vot.py \
    --speakers 0 --shards 6 --max-parallel 6 --out "$OUT"
echo "[task $SLURM_ARRAY_TASK_ID] done  $(date)"
