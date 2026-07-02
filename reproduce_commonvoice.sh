#!/usr/bin/env bash
# ============================================================================
# reproduce_commonvoice.sh — ONE-COMMAND reproduction of the Common Voice
# experiment, INCLUDING the corpus download.
#
#   ./reproduce_commonvoice.sh
#
# End to end: download+organize the study corpus from the committed manifest ->
# MFA-align -> extract the 40-feature table -> Dr.VOT VOT + merge ->
# speaker-count scaling analysis (PR / summed-MI / Fano; balanced pooled +
# gender-partition curves) -> report.
#
# PREREQUISITES (one-time — see SETUP.md):
#   * conda envs: voice-is-unique, deepformants, deepfry, drvot, aligner
#     (extractor/*.yml ; aligner via montreal-forced-aligner + english_us_arpa)
#   * A Hugging Face token with access to fsicoli/common_voice_{17,21,22}_0
#     (accept the dataset terms once), in $HF_TOKEN or ~/.cache/huggingface/token
#
# COMPUTE: this is a HEAVY, multi-day pipeline on a laptop (DeepFry+DeepFormants
# run per clip; Dr.VOT VOT is the long pole). For a cluster, use the turnkey
# SLURM job common-voice/run_commonvoice_psc.slurm and the psc_array/ arrays.
# Use PILOT=1 first to validate everything end to end in minutes.
#
# ENV OVERRIDES (all optional):
#   CV_OUT       rebuilt corpus dir        (default ~/data/commonvoice_repro)
#   CV_FEATS     feature table dir         (default ~/data/commonvoice-feats)
#   CV_VOT       Dr.VOT work dir           (default ~/data/cv-vot)
#   CV_ANALYSIS  analysis output dir       (default ~/data/cv-analysis)
#   JOBS         parallel workers          (default 8)
#   PILOT=1      first 20 speakers + 50-speaker VOT + tiny grid (fast smoke)
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/common-voice"

: "${CV_OUT:=$HOME/data/commonvoice_repro}"
: "${CV_FEATS:=$HOME/data/commonvoice-feats}"
: "${CV_VOT:=$HOME/data/cv-vot}"
: "${CV_ANALYSIS:=$HOME/data/cv-analysis}"
: "${JOBS:=8}"
: "${HF_TOKEN:=$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
export HF_TOKEN
PILOT="${PILOT:-}"

command -v conda >/dev/null || { echo "conda not found — see SETUP.md"; exit 1; }
[ -n "$HF_TOKEN" ] || { echo "no HF_TOKEN (needed to download Common Voice)"; exit 1; }
# resolve the real conda so DeepFormants/DeepFry/Dr.VOT subprocesses launch anywhere
export CV_CONDA="${CV_CONDA:-$(conda info --base)/bin/conda}"

echo "==> [1/4] download + organize + MFA-align the study corpus (from manifest/study8k)"
conda run -n voice-is-unique python scripts/reproduce_corpus.py \
    --manifest-dir manifest/study8k --out "$CV_OUT" --align --skip-classify \
    --workers "$JOBS" ${PILOT:+--limit 20}

echo "==> [2/4] extract the 40-feature table (DeepFormants + DeepFry per clip)"
conda run -n voice-is-unique python scripts/extract_feats_8k.py \
    --use-manifest manifest/feats_manifest_8000.tsv \
    --corpus "$CV_OUT" --wavroot "$CV_OUT/wavs" --out "$CV_FEATS" --jobs "$JOBS"

echo "==> [3/4] VOT via Dr.VOT + merge into the parquet (column #39)"
VOT_SPK=0; [ -n "$PILOT" ] && VOT_SPK=50   # 0 = all speakers; 50 = pilot
CV_CORPUS="$CV_OUT" CV_WAVROOT="$CV_OUT/wavs" \
conda run -n voice-is-unique python scripts/drvot_vot.py \
    --speakers "$VOT_SPK" --shards 28 --max-parallel 6 --out "$CV_VOT"
conda run -n voice-is-unique python scripts/merge_vot.py \
    --vot-table "$CV_VOT/vot_pilot.tsv" --parquet "$CV_FEATS/all_utterances.parquet"

echo "==> [4/4] speaker-count scaling analysis (PR / summed-MI / Fano)"
conda run -n voice-is-unique python scripts/run_scaling.py \
    --config CONFIG/common_voice.json --parquet "$CV_FEATS/all_utterances.parquet" \
    --out "$CV_ANALYSIS" ${PILOT:+--grid 100,250,500}

echo
echo "DONE (Common Voice). Deliverables:"
echo "  report : $CV_ANALYSIS/reports/scaling_report.txt"
echo "  tables : $CV_ANALYSIS/tables/scaling_*.csv"
echo "  figure : $CV_ANALYSIS/reports/figs/scaling_all.png"
