#!/bin/bash
# Bridges-2 batch job: download the exact 8k-study Common Voice clips, MFA-align them,
# and compute the 40 acoustic features. Submit with:  sbatch run_psc.slurm
# Prereq: run ../setup_psc.sh once, and huggingface-cli login (gated dataset).
#SBATCH -p RM                 # full regular-memory node: 128 cores, 256 GB
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -J cv_feats
#SBATCH -o cv_feats.%j.out
set -euo pipefail

GRANT=${GRANT:-<EDIT_ME>}                 # <-- set your allocation id (or `export GRANT=...`)
OCEAN=/ocean/projects/$GRANT/$USER
REPO=$OCEAN/vu/repo/voice-is-unique       # adjust if you cloned elsewhere
DATA=$OCEAN/vu/data/cv_study
VIU=$OCEAN/vu/envs/voice-is-unique

module load anaconda3
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
cd "$REPO/common-voice"

echo ">> Stage 1: download + decode + MFA-align (8k-study manifest)"
conda run -p "$VIU" python scripts/reproduce_corpus.py \
    --manifest-dir manifest/study8k --out "$DATA" --align --skip-classify

echo ">> Stage 2: 40-feature extraction (128-core node)"
conda run -p "$VIU" python scripts/extract_feats_8k.py \
    --use-manifest manifest/feats_manifest_8000.tsv \
    --corpus "$DATA" --wavroot "$DATA/wavs" --out "${DATA}-feats" --jobs 120

echo ">> Stage 3: verify speaker-level embeddings"
conda run -p "$VIU" python scripts/verify_speaker_embeddings.py \
    --config CONFIG/common_voice.json --parquet "${DATA}-feats/all_utterances.parquet"

echo ">> DONE. Features at ${DATA}-feats/all_utterances.parquet"
