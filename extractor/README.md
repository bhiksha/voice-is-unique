# timit-features — reproducible 40-feature utterance extractor for TIMIT

Computes 40 utterance-level acoustic features over the TIMIT corpus for the
*voice-is-unique* study (speaker distinctiveness). Masking is driven strictly by
the `.PHN` alignments (no energy VAD), uncomputable features are `NaN` (never
imputed), and runs are deterministic given the same input + CONFIG.

See `DECISIONS.md` for every pinned methodology choice and `CLAUDE.md` for the
project brief.

## Environments (conda)
- **`voice-is-unique`** — analysis env (numpy/scipy/librosa/soundfile/pandas/
  pyarrow/praat-parselmouth); torch-free. `environment.yml`.
- **`deepformants`** — DeepFormants F1–F4 (PyTorch CPU). `environment.deepformants.yml`.
- **`deepfry`** — DeepFry creak detector for VFI (py3.8/torch 1.12). `environment.deepfry.yml`.

DeepFormants and DeepFry are vendored under `third_party/` and invoked as
subprocesses, so torch never enters the analysis env. Also needs the system `sox`.

## Run
```
conda run -n voice-is-unique python -m timit_features.cli <TIMIT_ROOT> \
    --out ~/data/timit-feats --jobs 8
```
Flags: `--utt PATH` (one file), `--speaker ID`, `--limit N`, `--jobs N`,
`--overwrite`. The run is **resumable** (skips utterances whose JSON exists) and
**error-isolated** (a bad utterance → all-NaN record, logged, never aborts).

## Outputs (under `--out`)
- `TRAIN|TEST/DR*/<speaker>/<basename>.json` — per-utterance ids + 40 features + per-feature coverage.
- `all_utterances.parquet` / `.csv` — one row per utterance (ids + 40 features + `cov_*`).
- `CONFIG.json` — full parameter set (+ hash, stamped into every record).
- `MANIFEST.json` — seen/decoded/failed counts, decode-failure list, per-speaker counts, per-feature coverage.

## QA summary
```
conda run -n voice-is-unique python -m timit_features.report ~/data/timit-feats
```
Reports per-feature coverage, distribution, and the between/within-speaker
variance ratio (an ICC-like speaker-distinctiveness index).

## The 40 features
Source/glottal (IAIF): F0, jitter, shimmer, GCT, CQ, MFDR, SQ, NAQ, SHR, IHI,
VFI, semitone_SD_F0, GNE, CPP, dCPP. Formants (DeepFormants F1–F4 + order-20 Burg
F5/bandwidths): F1–F5, B1–B5. Resonance: Nasality (A1–P0). Spectral/energy:
spectral_skewness/kurtosis/entropy/rolloff/flux, alpha_ratio, LHR, SPI, SSPF,
RMS, AMD. Alignment-native: speech_rate, VOT, BGD. Order/domain/aggregation are
fixed in `src/timit_features/config.py`.

## Tests
```
conda run -n voice-is-unique python -m pytest
```
(Corpus- and model-env-dependent tests skip automatically when those are absent.)
