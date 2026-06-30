#!/bin/bash
# One-shot: build chunks, then submit the two arrays + their finalizers with the right
# dependencies. Feature and VOT arrays run concurrently (both read-only on the wavs);
# the final merge waits for the assembled parquet AND the VOT array.
#
#   feats_array[0-79] --afterok--> feats_assemble --\
#                                                    >--afterok--> vot_finalize (merge_vot)
#   vot_array[0-79]   ------------------------------/
#
# Run from this directory on a Bridges login node AFTER the wavs are built and the
# voice-is-unique/deepformants/deepfry/drvot envs are unpacked.
set -euo pipefail
O=/ocean/projects/cis250019p/ramakrib
REPO=$O/vu/repo/voice-is-unique/common-voice
N=${N:-80}                       # array size (chunks). Override: N=120 ./submit_all.sh
cd "$REPO/psc_array"
mkdir -p logs

echo "### building $N chunks ###"
bash make_chunks.sh "$REPO/manifest/feats_manifest_8000.tsv" "$N" "$O/vu/chunks"

echo "### submitting feature array ###"
F=$(sbatch --parsable --array=0-$((N-1)) feats_array.slurm)
echo "  feats_array  = $F"
FA=$(sbatch --parsable --dependency=afterok:$F feats_assemble.slurm)
echo "  feats_assemble = $FA  (afterok:$F)"

echo "### submitting VOT array ###"
V=$(sbatch --parsable --array=0-$((N-1)) vot_array.slurm)
echo "  vot_array    = $V"
VF=$(sbatch --parsable --dependency=afterok:$FA:$V vot_finalize.slurm)
echo "  vot_finalize = $VF  (afterok:$FA:$V)"

echo
echo "submitted. watch:  squeue -u \$USER"
echo "results when done: $O/vu/data/cv_feats/all_utterances.parquet (with VOT)"
