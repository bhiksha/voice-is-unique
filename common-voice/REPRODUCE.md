# Reproducing the Common Voice corpus from the manifest

The Common Voice corpus this project analyses (12,971 English speakers × 10 clips ≥ 4 s,
gender-branched, with the "unknown" speakers gender-estimated by classifier confidence) was
assembled incrementally by the pipeline in `~/data` (`cv_*.py`, `gender_classify.py`). Audio
cannot be redistributed, so instead we commit a **manifest** — the exact list of clips and
gender decisions — plus a script that re-downloads, re-organizes, and re-classifies to
reconstruct the corpus bit-for-bit.

## What's committed

| File | Rows | What it is |
|------|------|------------|
| `manifest/corpus_clips.tsv` | 129,710 | **the list of all downloaded files**: `speaker_dir, clip, release, split`. `release` is the HF dataset id each clip's audio is fetched from (CV17/21/22); `split ∈ {train, other}`. |
| `manifest/corpus_speakers.tsv` | 12,971 | per speaker: `speaker_dir, gender_dir, client_id, n_clips, gender_source, decision, majority_count, avg_conf`. `gender_source` is `self_reported` (female/male dirs) or `estimated` (unknown dir); for estimated speakers the `decision` (`tagged_female`/`tagged_male`/`ambiguous`/`tie`) etc. come from the wav2vec2 classifier. |

Corpus composition recorded by the manifest: **2,099** self-reported female + **5,006** male +
**5,866** unknown → gender-estimated as **1,911** tagged_female, **3,700** tagged_male, 219
ambiguous, 36 tie. **Female total = 2,099 + 1,911 = 4,010** (CV-English physical ceiling ≈ 4,340).
Clips resolve almost entirely to CV22 (129,534), with a small tail from CV21 (33) and CV17 (143).

## Scripts (`scripts/`)

- **`reproduce_corpus.py`** — rebuilds the corpus from the manifest. Phases (each resumable):
  1. **download + organize** — streams each release/split's audio tars, extracts only the
     manifest's clips, writes the gender-branched mp3 tree (`<out>/<gender>/<grp>/<sd>/<clip>.mp3`),
     a flat 16 kHz wav tree (`<out>/wavs/<sd>/<clip>.wav`, for classify/MFA), and `speaker_map.tsv`.
  2. **align** *(optional, `--align`)* — MFA force-align → `phone_segments.parquet` (env `aligner` + sox).
  3. **classify** — runs `gender_classify.py` over the unknown speakers' wavs → `unknown/estimated_gender`,
     then **verifies** the freshly-estimated decisions against the frozen manifest and reports drift.
- **`gender_classify.py`** — vendored from the `~/data` pipeline; identical logic, with the corpus
  and wav roots taken from `CV_ROOT` / `CV_WAVROOT` (set automatically by `reproduce_corpus.py`).
- **`build_manifest.py`** — how the manifest above was generated from a live corpus (resolves each
  clip's release/split against the cached CV metadata). You don't need this to reproduce; it's the
  provenance of the manifest.

## How to run

Common Voice on Hugging Face is **gated** — accept the dataset terms for
`fsicoli/common_voice_{17,21,22}_0` with your HF account and have a token first.

```bash
conda activate voice-is-unique
export HF_TOKEN=...                      # or ~/.cache/huggingface/token

# full rebuild (download + organize + classify), into a fresh dir:
python scripts/reproduce_corpus.py --out ~/data/commonvoice_repro --workers 4

# smoke test on the first few speakers:
python scripts/reproduce_corpus.py --out /tmp/repro_test --limit 5

# also rebuild MFA phone alignments:
python scripts/reproduce_corpus.py --out ~/data/commonvoice_repro --align
```

The classify phase runs in the **`gender-id`** conda env (CPU torch + transformers); the script
invokes that interpreter itself, so you only need to be in `voice-is-unique` to launch it.
`--out` defaults to `~/data/commonvoice_repro` so a rebuild never clobbers the original
`~/data/commonvoice`.

## Practical notes

- **Use `--workers 4` for classification.** The 8-worker run has a documented
  `multiprocessing.Pool` OOM-stall on the last few speakers (machine has 8 cores / ~15 GB).
  If it ever hangs on the tail, kill it and re-run — it resumes from `estimated_gender`.
- **Download time** is dominated by tar streaming. CV's HF mirror has no per-clip index, so a
  split's shards are streamed (newest-first) until the wanted clips are found. A *full* rebuild
  is efficient (every shard yields many hits); a *tiny* subset of old clips is the worst case
  (you may stream several shards before a hit). The full corpus is ~130k clips.
- **Reproducibility.** Re-downloaded audio is byte-identical, and the classifier is deterministic
  per clip, so the verify step should report ~0 drift. (A handful of borderline speakers near the
  9/10 tagging threshold could in principle flip; the manifest's frozen decisions are the
  reference.)
- This rebuilds the **corpus**. Downstream speaker-distinctiveness analysis (PR / summed-MI / Fano)
  then runs on it per the project `README.md`.
