#!/bin/bash
# Split a feats_manifest (header + one row per speaker) into N balanced chunks for a
# SLURM array. Round-robin by row index so every chunk has ~equal speakers/clips.
# Usage: make_chunks.sh <manifest.tsv> <N> <outdir>
set -euo pipefail
MAN=${1:?manifest}; N=${2:?nchunks}; OUT=${3:?outdir}
mkdir -p "$OUT"
hdr=$(head -1 "$MAN")
tail -n +2 "$MAN" | awk -v n="$N" -v out="$OUT" -v hdr="$hdr" '
  BEGIN { for (i=0;i<n;i++){ f=sprintf("%s/chunk_%03d.tsv",out,i); print hdr > f } }
  { f=sprintf("%s/chunk_%03d.tsv",out,(NR-1)%n); print >> f }
'
total=$(( $(wc -l < "$MAN") - 1 ))
echo "wrote $N chunks to $OUT ($total speakers; ~$((total/N)) per chunk)"
