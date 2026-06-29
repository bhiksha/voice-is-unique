#!/bin/bash
# Bridges-2 batch job: the ENTIRE Common Voice experiment, end to end:
#   download (exact 8k-study clip list) -> MFA-align -> 40-feature extraction
#   -> Dr.VOT (voiceless P/T/K) merged into the VOT feature -> verify embeddings
#   -> speaker-count scaling analysis (balanced N=1000..8000) + gender-specific analyses.
# Submit:  cd common-voice && export GRANT=<id> && sbatch run_commonvoice_psc.slurm
#
# Prereq: run ../setup_psc.sh once, and `conda run -n voice-is-unique huggingface-cli login`
# (Common Voice is gated). Every stage is resumable, so re-submitting continues where it left off.
#SBATCH -p RM                 # 128 cores, 256 GB
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J cv_full
#SBATCH -o cv_full.%j.out
set -euo pipefail

GRANT=${GRANT:-<EDIT_ME>}
OCEAN=/ocean/projects/$GRANT/$USER
REPO=$OCEAN/vu/repo/voice-is-unique
DATA=$OCEAN/vu/data/cv_study          # corpus tree + wavs + phone_segments.parquet
FEATS=${DATA}-feats                   # per-clip JSONs + all_utterances.parquet
VOT=$OCEAN/vu/data/cv_vot             # Dr.VOT working dir + vot_pilot.tsv
ANALYSIS=$OCEAN/vu/data/cv-analysis
module load anaconda3
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
cd "$REPO/common-voice"

echo ">> [1/6] download exact 8k-study clips + decode + MFA-align"
conda run -n voice-is-unique python scripts/reproduce_corpus.py \
    --manifest-dir manifest/study8k --out "$DATA" --align --skip-classify

echo ">> [2/6] 40-feature extraction (128-core node)"
conda run -n voice-is-unique python scripts/extract_feats_8k.py \
    --use-manifest manifest/feats_manifest_8000.tsv \
    --corpus "$DATA" --wavroot "$DATA/wavs" --out "$FEATS" --jobs 120

echo ">> [3/6] Dr.VOT on voiceless stops (sharded; big-memory node -> many shards)"
CV_CORPUS="$DATA" CV_WAVROOT="$DATA/wavs" \
conda run -n voice-is-unique python scripts/drvot_vot.py \
    --speakers 0 --shards 64 --max-parallel 32 --out "$VOT"

echo ">> [4/6] merge Dr.VOT VOT into the features parquet (fills the VOT column)"
conda run -n voice-is-unique python scripts/merge_vot.py \
    --vot-table "$VOT/vot_pilot.tsv" --parquet "$FEATS/all_utterances.parquet"

echo ">> [5/6] verify speaker-level embeddings"
conda run -n voice-is-unique python scripts/verify_speaker_embeddings.py \
    --config CONFIG/common_voice.json --parquet "$FEATS/all_utterances.parquet"

echo ">> [6/6] scaling analysis: balanced N=1000..8000 (pooled + within-sex) + female-only + male-only"
conda run -n voice-is-unique python scripts/run_scaling.py \
    --parquet "$FEATS/all_utterances.parquet" \
    --grid 500,1000,1500,2000,2500,3000,3500,4000 --out "$ANALYSIS"

echo ">> DONE. features: $FEATS ; VOT: $VOT/vot_pilot.tsv ; analysis: $ANALYSIS/tables, reports/figs"
