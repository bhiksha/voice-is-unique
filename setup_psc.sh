#!/bin/bash
# One-time PSC Bridges-2 setup for BOTH experiments (TIMIT + Common Voice).
# Builds the conda envs (NAMED, stored in Ocean so they don't fill the 25 GB $HOME)
# and downloads the MFA models. The envs are NAMED on purpose: the extractor shells out
# to `conda run -n deepformants` / `-n deepfry` internally, which needs name resolution.
# Run on a LOGIN node:   GRANT=<your_allocation_id> bash setup_psc.sh
set -euo pipefail
: "${GRANT:?set GRANT=<your_allocation_id> (see the 'projects' command)}"
OCEAN=/ocean/projects/$GRANT/$USER
REPO="$(cd "$(dirname "$0")" && pwd)"

echo ">> Ocean base: $OCEAN/vu   repo: $REPO"
mkdir -p "$OCEAN"/vu/{data,envs,pkgs}
module load anaconda3
# put envs + package cache in Ocean, and make Ocean the FIRST envs dir so `-n` envs land there
conda config --add pkgs_dirs "$OCEAN/vu/pkgs"
conda config --add envs_dirs "$OCEAN/vu/envs"

echo ">> [1/5] voice-is-unique (analysis + extractor)"
conda env create -n voice-is-unique -f "$REPO/extractor/environment.yml"
conda run -n voice-is-unique pip install -e "$REPO/extractor"   # provides `timit-features` + timit_features pkg
echo ">> [2/5] deepformants (F1-F4)"
conda env create -n deepformants -f "$REPO/extractor/environment.deepformants.yml"
echo ">> [3/5] deepfry (VFI)"
conda env create -n deepfry -f "$REPO/extractor/environment.deepfry.yml"
echo ">> [4/5] drvot (VOT, Common Voice only)"
conda env create -n drvot -f "$REPO/extractor/environment.drvot.yml"
echo ">> [5/5] aligner (MFA) + english_us_arpa models (Common Voice only)"
conda create -n aligner -c conda-forge montreal-forced-aligner -y
conda run -n aligner mfa model download acoustic   english_us_arpa
conda run -n aligner mfa model download dictionary english_us_arpa

# make sure the vendored Dr.VOT feature binary is executable after a fresh clone
chmod +x "$REPO/extractor/third_party/drvot/process_data/linux_VotFrontEnd2" 2>/dev/null || true

cat <<EOF

Setup complete. Named envs in $OCEAN/vu/envs : voice-is-unique, deepformants, deepfry, drvot, aligner
For Common Voice (gated dataset):
  1) accept terms on huggingface.co (fsicoli/common_voice_22_0, _21_0, _17_0)
  2) conda run -n voice-is-unique huggingface-cli login    # paste your HF token
Then submit jobs:
  cd $REPO/common-voice
  export GRANT=$GRANT
  sbatch run_commonvoice_psc.slurm     # full CV: download -> features+VOT -> analysis
  sbatch run_timit_psc.slurm           # TIMIT: features -> analysis  (set TIMIT_ROOT first)
EOF
