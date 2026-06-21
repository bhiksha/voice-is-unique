# voice-is-unique / timit-features — project brief for Claude

## What this is
A **reproducible acoustic feature-extraction pipeline** for the TIMIT corpus,
producing **40 utterance-level features** for a study of speaker distinctiveness.
The within-speaker variance of each feature is load-bearing scientifically, so
masking and parameter choices must be exact and documented — never guessed.

The authoritative spec is the prompt at:
`~/Dropbox/docs/papers/2025/voice-is-unique/PNAS_nexus/CLAUDE_CODE_PROMPT_timit_features.md`

## Hard rules (from the prompt — do not violate)
1. **No invented methodology.** Every number-affecting parameter lives in the
   CONFIG block (`src/timit_features/config.py`). If unsure of a value, STOP and
   ask; never bury choices in function bodies.
2. **Never impute.** Uncomputable feature → `NaN`. Missingness is real signal.
3. **Mask non-speech via `.PHN` alignments**, not energy VAD. Each feature is
   computed only on its phone-class domain. `h#`/`pau`/`epi` are silence.
4. **Determinism.** Fixed seeds; same input + same CONFIG → byte-identical output.
5. **Verify before scale.** Show CONFIG → run one utterance → run one speaker →
   self-checks, pausing for approval at each. Do NOT process the full corpus until
   the user explicitly says **"create the feats."**

## Environment
- Analysis env **`voice-is-unique`** (Python 3.11), separate from `audiogenie`.
  Interpreter: `~/miniconda3/envs/voice-is-unique/bin/python`. **torch-free** by
  design. Key packages: numpy, scipy, librosa, soundfile (pip), pandas, pyarrow,
  praat-parselmouth (pip), tqdm, pytest (`environment.yml`).
- Isolated env **`deepformants`** (`environment.deepformants.yml`): numpy, scipy,
  conda-forge `pytorch-cpu`. Runs the vendored DeepFormants F1–F4 estimator ONLY,
  invoked as a subprocess so torch never touches the analysis env. Also needs the
  system `sox` binary.
- Isolated env **`deepfry`** (`environment.deepfry.yml`): Python 3.8 / torch 1.12
  CPU (`mkl=2024.0.0` pinned). Runs the vendored DeepFry creak detector for VFI
  (#11), invoked as a subprocess. Vendored at `third_party/deepfry/`.
- libsndfile (via soundfile) and parselmouth both decode TIMIT NIST SPHERE
  `.WAV` directly at 16 kHz — verified.

## Formant estimator (hybrid — DECISIONS #D)
- **F1–F4** from vendored **DeepFormants** (PyTorch) in `third_party/deepformants/`
  (pinned commit, patched for numpy2/scipy1.17, wandb-free wrapper `df_infer.py`).
  Feasibility proven: deterministic, sane formants on TIMIT vowels. See VENDOR.md.
- **F5 + B1–B5** from order-20 Burg LPC at native 16 kHz, with spurious poles
  rejected by matching to the DeepFormants F1–F4 (not yet implemented).

## Data
- TIMIT root (read-only): `~/data/timit/TIMIT` (children `TRAIN/`, `TEST/`;
  `DR1`..`DR8`; speaker dirs `<sex><id>/`; per-utterance `.WAV/.PHN/.WRD/.TXT`).
- 61-phone TIMIT ARPABET set; `.PHN`/`.WRD` use sample indices at 16 kHz.

## Status
- CONFIG block approved (checkpoint 1). **All 40 features implemented** and
  verified end-to-end on one utterance (checkpoint 2) and one speaker (FCJF0,
  checkpoint 3). Protocol self-checks pass (silence→NaN, no-voiced→voiced NaN,
  coverage consistency, determinism). 46 tests green.
- CLI: `timit-features <root> [--out DIR] [--limit N] [--speaker ID] [--utt PATH]
  [--jobs N]`. Outputs per-utterance JSON (mirrors TIMIT), all_utterances.parquet
  /.csv, CONFIG.json, MANIFEST.json. ~12 s/utterance (DeepFormants + DeepFry
  subprocesses) → use --jobs for the corpus.
- **Awaiting "create the feats"** before the full-corpus run.
- VFI = DeepFry (env `deepfry`); Nasality = A1-P0 signal processing (in-env).
  Operational items still flagged for review in DECISIONS (IAIF CQ/SQ geometry,
  IHI/SHR/AMD, jitter/shimmer per-segment, linear-vs-dB band ratios).

## Layout
```
voice-is-unique/
  environment.yml        conda env spec
  pyproject.toml
  DECISIONS.md           open methodology questions awaiting sign-off
  src/timit_features/
    config.py            THE CONFIG BLOCK (params, phone→class map, 40-feature
                         order/domain/aggregation, config hash)
    __init__.py
  tests/
```
