#!/bin/bash
# One-time PSC Bridges-2 setup for the Common Voice download + 40-feature pipeline.
# Builds the conda envs (into Ocean, not the 25 GB $HOME) and downloads the MFA models.
# Run on a LOGIN node:   GRANT=<your_allocation_id> bash setup_psc.sh
set -euo pipefail
: "${GRANT:?set GRANT=<your_allocation_id> (see the 'projects' command)}"
OCEAN=/ocean/projects/$GRANT/$USER
REPO="$(cd "$(dirname "$0")" && pwd)"

echo ">> Ocean base: $OCEAN/vu   repo: $REPO"
mkdir -p "$OCEAN"/vu/{data,envs,pkgs}
module load anaconda3
conda config --add envs_dirs "$OCEAN/vu/envs"
conda config --add pkgs_dirs "$OCEAN/vu/pkgs"

echo ">> [1/4] voice-is-unique env (analysis + extractor)"
conda env create -p "$OCEAN/vu/envs/voice-is-unique" -f "$REPO/extractor/environment.yml"
echo ">> [2/4] deepformants env (F1-F4)"
conda env create -p "$OCEAN/vu/envs/deepformants" -f "$REPO/extractor/environment.deepformants.yml"
echo ">> [3/4] deepfry env (VFI)"
conda env create -p "$OCEAN/vu/envs/deepfry" -f "$REPO/extractor/environment.deepfry.yml"
echo ">> [4/4] aligner env (MFA) + english_us_arpa models"
conda create -p "$OCEAN/vu/envs/aligner" -c conda-forge montreal-forced-aligner -y
conda run -p "$OCEAN/vu/envs/aligner" mfa model download acoustic   english_us_arpa
conda run -p "$OCEAN/vu/envs/aligner" mfa model download dictionary english_us_arpa

cat <<EOF

Setup complete. Envs are in $OCEAN/vu/envs.
Next:
  1) Accept the Common Voice terms on huggingface.co (fsicoli/common_voice_22_0, _21_0, _17_0)
  2) conda run -p $OCEAN/vu/envs/voice-is-unique huggingface-cli login   # paste your HF token
  3) cd $REPO/common-voice && edit GRANT in run_psc.slurm && sbatch run_psc.slurm
EOF
