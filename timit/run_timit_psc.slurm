#!/bin/bash
# Bridges-2 batch job: the entire TIMIT experiment = 40-feature extraction + analysis
# (PR / summed-MI / Fano / Fisher report). Submit:  sbatch run_timit_psc.slurm
#
# Prereq: run ../setup_psc.sh once. TIMIT is LDC-licensed -- bring your own copy and set
# TIMIT_ROOT to the directory that contains TRAIN/ and TEST/.
#SBATCH -p RM                 # 128 cores, 256 GB
#SBATCH -N 1
#SBATCH -t 8:00:00
#SBATCH -J timit
#SBATCH -o timit.%j.out
set -euo pipefail

GRANT=${GRANT:-<EDIT_ME>}
TIMIT_ROOT=${TIMIT_ROOT:?set TIMIT_ROOT to your TIMIT corpus (has TRAIN/ TEST/); LDC-licensed}
OCEAN=/ocean/projects/$GRANT/$USER
REPO=$OCEAN/vu/repo/voice-is-unique
FEATS=$OCEAN/vu/data/timit-feats
ANALYSIS=$OCEAN/vu/data/timit-analysis
module load anaconda3

echo ">> Stage 1: 40-feature extraction (128-core node; shells to deepformants/deepfry)"
conda run -n voice-is-unique timit-features "$TIMIT_ROOT" --out "$FEATS" --jobs 120

echo ">> Stage 2: analysis (PR / summed-MI / Fano + report)"
cd "$REPO/timit"
conda run -n voice-is-unique python - <<PY
import json, pathlib
c = json.load(open("CONFIG/timit.json"))
c["input_parquet"] = "$FEATS/all_utterances.parquet"
pathlib.Path("CONFIG/timit_psc.json").write_text(json.dumps(c, indent=2))
PY
mkdir -p "$ANALYSIS"
conda run -n voice-is-unique python -m src.run_all --config CONFIG/timit_psc.json --out "$ANALYSIS"

echo ">> DONE. features: $FEATS/all_utterances.parquet ; analysis: $ANALYSIS/"
