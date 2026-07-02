#!/usr/bin/env bash
# ============================================================================
# reproduce_timit.sh — ONE-COMMAND reproduction of the TIMIT experiment.
#
#   ./reproduce_timit.sh
#
# Produces the TIMIT speaker-distinctiveness result (PR / summed-MI / Fano,
# pooled + within-sex) from a TIMIT corpus you supply.
#
# PREREQUISITES (one-time — see SETUP.md):
#   * conda envs: voice-is-unique, deepformants, deepfry  (extractor/*.yml)
#   * TIMIT audio from the LDC (LDC93S1) at $TIMIT_ROOT  (contains TRAIN/ TEST/)
#     — TIMIT is licensed and is NOT redistributed with this repo.
#
# ENV OVERRIDES (all optional):
#   TIMIT_ROOT   TIMIT corpus root         (default ~/data/timit)
#   TIMIT_FEATS  where the feature table goes (default ~/data/timit-feats)
#   JOBS         parallel workers          (default 8)
#   PILOT=1      quick 200-utterance smoke test (validate the pipeline fast)
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${TIMIT_ROOT:=$HOME/data/timit}"
: "${TIMIT_FEATS:=$HOME/data/timit-feats}"
: "${JOBS:=8}"
PILOT="${PILOT:-}"
PARQUET="$TIMIT_FEATS/all_utterances.parquet"

command -v conda >/dev/null || { echo "conda not found — see SETUP.md"; exit 1; }
# make DeepFormants/DeepFry resolvable regardless of where conda is installed
export CV_CONDA="${CV_CONDA:-$(conda info --base)/bin/conda}"

echo "==> [1/2] extract the 40-feature table from TIMIT at $TIMIT_ROOT"
if [ -f "$PARQUET" ] && [ -z "$PILOT" ]; then
  echo "    reusing existing $PARQUET (delete it to force re-extraction)"
else
  [ -d "$TIMIT_ROOT" ] || { echo "TIMIT audio not found at TIMIT_ROOT=$TIMIT_ROOT (obtain from LDC)"; exit 1; }
  ( cd "$ROOT/extractor" && PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
    conda run -n voice-is-unique python -m timit_features.cli "$TIMIT_ROOT" \
      --out "$TIMIT_FEATS" --jobs "$JOBS" ${PILOT:+--limit 200} )
fi

echo "==> [2/2] run the TIMIT analysis (PR / summed-MI / Fano; pooled + within-sex)"
( cd "$ROOT/timit" && pip install -q -r requirements.txt \
  && python -m src.run_all --config CONFIG/timit.json ${PILOT:+--pilot} )

echo
echo "DONE (TIMIT). Deliverables:"
echo "  report : $ROOT/timit/reports/report.md"
echo "  tables : $ROOT/timit/tables/*.csv (+ provenance.json)"
echo "  figures: $ROOT/timit/reports/figs/"
echo
echo "For the cross-corpus PARITY scaling variant (same code as Common Voice), see"
echo "  common-voice/psc_array/timit_scaling.slurm  (uses CONFIG/timit.json)."
