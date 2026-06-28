# Reproducing the Common Voice 8,000-speaker feature study (PNAS)

This is the **end-to-end repeatability package** for the study's Common Voice arm: a balanced
**4,000 female + 4,000 male** speaker set, 10 clips each (**80,000 clips**), with the **40-feature**
acoustic representation computed per clip and averaged to speaker-level embeddings.

Common Voice audio cannot be redistributed, so the repo commits the **exact clip manifest** (what
to download) plus **scripts that re-download and re-extract** to reconstruct the features bit-for-bit.

## Committed manifests (`manifest/`)

| File | Rows | Role |
|------|------|------|
| **`study8k_clips.tsv`** | 80,000 | **the exact set of files used**: `speaker_dir, sex, gender_source, clip, release, split`. `release` = the HF dataset each clip's audio is fetched from; `split ∈ {train, other}`. |
| **`feats_manifest_8000.tsv`** | 8,000 | the frozen speaker set + **sex label** (drives extraction; no re-classification, so labels are exact). |
| `study8k/corpus_clips.tsv` | 80,000 | same clips, in the schema `reproduce_corpus.py` consumes. |
| `study8k/corpus_speakers.tsv` | 8,000 | per-speaker `gender_dir, client_id, rel_path` for tree layout + MFA. |

Composition (frozen): 40,000 F + 40,000 M clips; releases 79,868 CV22 / 25 CV21 / 107 CV17;
splits 74,595 train / 5,405 other. Sex per speaker is self-reported (`m_*`,`f_*` dirs) or, for the
female arm's `u_*` speakers, the wav2vec2 classifier tag — **frozen here so reproduction needs no
re-classification**.

## Environments

Conda envs under `~/miniconda3/envs/` (specs in `vu-pkg/extractor/environment*.yml`):
`voice-is-unique` (orchestration + the 40-feature extractor, torch-free), `aligner`
(Montreal Forced Aligner + `english_us_arpa` dictionary & acoustic model), `deepformants` and
`deepfry` (the F1–F4 and VFI sub-extractors, invoked as subprocesses), plus `sox`/`ffmpeg`.
Common Voice on Hugging Face is **gated**: accept the dataset terms for
`fsicoli/common_voice_{17,21,22}_0` and export `HF_TOKEN` before step 1.

## Exact reproduction (3 steps)

```bash
conda activate voice-is-unique
export HF_TOKEN=...                      # gated CV access
cd ~/claude/vu-pkg/common-voice
D=~/data/cv_study_repro                  # any fresh working dir

# 1) DOWNLOAD the exact clips, organize the speaker tree, decode 16 kHz wavs, MFA-align
#    -> $D/wavs/<sd>/*.wav + $D/phone_segments.parquet + $D/speaker_map.tsv
#    (--skip-classify: sex labels are frozen in the manifest, so no wav2vec2 step is needed)
python scripts/reproduce_corpus.py --manifest-dir manifest/study8k --out "$D" --align --skip-classify

# 2) EXTRACT the 40 features for the exact 8,000 study speakers (frozen sex labels)
#    -> $D-feats/<gender>/<grp>/<sd>/<stem>.json + $D-feats/all_utterances.parquet
python scripts/extract_feats_8k.py --use-manifest manifest/feats_manifest_8000.tsv \
       --corpus "$D" --wavroot "$D/wavs" --out "$D-feats"

# 3) VERIFY speaker-level embeddings (per-speaker mean of finite clips)
python scripts/verify_speaker_embeddings.py --config CONFIG/common_voice.json \
       --parquet "$D-feats/all_utterances.parquet"
```

Each step is **resumable** (re-run to continue after an interruption): downloads/alignments/JSONs
that already exist are skipped. Step 1 streams HF tars; step 2 spawns a DeepFormants and a DeepFry
model-load subprocess per clip (~14 s/clip → the full 80k run is ~2 days at `--jobs 6` on 8 cores).

## Expected result (what we got)

- 80,000 clips extracted, **100% decode_ok**; `all_utterances.parquet` = 80,000 × 85 (40 features +
  40 coverage + metadata).
- **39 of 40 features are usable.** `VOT` is NaN for every clip: MFA emits one segment per stop with
  no separate closure, so Voice Onset Time is uncomputable from MFA alignments (a documented
  cross-corpus limitation, not a bug).
- Speaker embeddings (per-speaker mean of finite clips): **8,000 / 8,000 speakers have a complete
  embedding over all 39 usable features — zero per-speaker failures**; the only global drop is VOT.
  `verify_speaker_embeddings.py` writes `speaker_embeddings.parquet`, `embedding_failures.tsv`, and
  `embedding_verify_report.txt`.

## Determinism / provenance

- Same manifest + same CV snapshot + fixed CONFIG (`CONFIG/common_voice.json`, with the extractor's
  CONFIG hash) → identical features; re-downloaded mp3s are byte-identical and the extractor is
  deterministic (fixed seeds, single-threaded BLAS per worker).
- How the manifests were produced from the live corpus is recorded in `scripts/build_manifest.py`
  (corpus list) and `scripts/extract_feats_8k.py` (the deterministic 8k selection rule). The corpus
  build itself is documented in `~/data/CLAUDE.md` and `REPRODUCE.md`.
- The corpus's MFA phone alignments are regenerated in step 1 (`--align`); the study does not depend
  on any pre-existing working tree.
