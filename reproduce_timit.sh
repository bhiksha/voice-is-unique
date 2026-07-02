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
#   TIMIT_ROOT   TIMIT corpus root            (default ~/data/timit)
#   TIMIT_FEATS  where the feature table goes (default ~/data/timit-feats)
#   TIMIT_OUT    where tables/reports go      (default ./timit)
#   JOBS         parallel workers             (default 8)
#   PILOT=1      quick smoke test: extract only PILOT_LIMIT utts and write
#                everything to a scratch dir (never touches your real data)
#   PILOT_LIMIT  utterances for the pilot     (default 200)
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PILOT="${PILOT:-}"

: "${TIMIT_ROOT:=$HOME/data/timit}"
: "${JOBS:=8}"
: "${PILOT_LIMIT:=200}"
if [ -n "$PILOT" ]; then          # pilot: isolate to scratch, never clobber real data
  : "${TIMIT_FEATS:=${TMPDIR:-/tmp}/vu_timit_pilot/feats}"
  : "${TIMIT_OUT:=${TMPDIR:-/tmp}/vu_timit_pilot/out}"
else
  : "${TIMIT_FEATS:=$HOME/data/timit-feats}"
  : "${TIMIT_OUT:=$ROOT/timit}"
fi
PARQUET="$TIMIT_FEATS/all_utterances.parquet"

command -v conda >/dev/null || { echo "conda not found — see SETUP.md"; exit 1; }
# make DeepFormants/DeepFry resolvable regardless of where conda is installed
export CV_CONDA="${CV_CONDA:-$(conda info --base)/bin/conda}"

echo "==> [1/2] extract the 40-feature table from TIMIT at $TIMIT_ROOT  ->  $TIMIT_FEATS"
if [ -f "$PARQUET" ] && [ -z "$PILOT" ]; then
  echo "    reusing existing $PARQUET (delete it to force re-extraction)"
else
  [ -d "$TIMIT_ROOT" ] || { echo "TIMIT audio not found at TIMIT_ROOT=$TIMIT_ROOT (obtain from LDC)"; exit 1; }
  ( cd "$ROOT/extractor" && PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
    conda run -n voice-is-unique python -m timit_features.cli "$TIMIT_ROOT" \
      --out "$TIMIT_FEATS" --jobs "$JOBS" ${PILOT:+--limit "$PILOT_LIMIT"} )
fi

echo "==> [2/2] run the TIMIT analysis (PR / summed-MI / Fano; pooled + within-sex)  ->  $TIMIT_OUT"
# run_all reads input_parquet FROM the config, so derive a config pointing at the
# parquet we just built (lets TIMIT_FEATS be anywhere; keeps the pilot self-contained).
RUN_CFG="$TIMIT_FEATS/_run_config.json"
conda run -n voice-is-unique python - "$ROOT/timit/CONFIG/timit.json" "$PARQUET" "$RUN_CFG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1])); cfg["input_parquet"] = sys.argv[2]
json.dump(cfg, open(sys.argv[3], "w"), indent=2)
PY
mkdir -p "$TIMIT_OUT"
( cd "$ROOT/timit" && pip install -q -r requirements.txt \
  && python -m src.run_all --config "$RUN_CFG" --out "$TIMIT_OUT" )

echo
echo "DONE (TIMIT). Deliverables under $TIMIT_OUT:"
echo "  report : $TIMIT_OUT/reports/report.md"
echo "  tables : $TIMIT_OUT/tables/*.csv (+ provenance.json)"
echo "  figures: $TIMIT_OUT/reports/figs/"
echo
echo "Cross-corpus PARITY scaling variant (same code as Common Voice):"
echo "  common-voice/psc_array/timit_scaling.slurm  (uses CONFIG/timit.json)."
