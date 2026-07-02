# voice-is-unique

How much **speaker identity** is carried by a fixed 40-feature, utterance-level
acoustic representation of voice — measured three ways (effective dimensionality,
an upper bound, and a lower bound on joint speaker information) and shown to scale
with the number of speakers rather than saturating at a voice-imposed limit.

## Reproduce (one command per experiment)

After a one-time setup — conda envs + data + a Hugging Face token, see **[SETUP.md](SETUP.md)**:

```bash
# TIMIT (reference experiment) — supply TIMIT audio from the LDC at $TIMIT_ROOT
./reproduce_timit.sh

# Common Voice (INCLUDES the corpus download from Hugging Face)
./reproduce_commonvoice.sh
```

Each script runs its whole pipeline end to end and writes a report + tables + figures:
extraction → analysis (TIMIT); download → MFA-align → 40-feature extract → Dr.VOT VOT →
speaker-count scaling analysis (Common Voice). Both are **resumable** and **deterministic**
(fixed seeds). Prepend **`PILOT=1`** to validate the entire pipeline in minutes on a small
subset before committing to the full (multi-day) run. On an HPC cluster, use the turnkey
`common-voice/run_commonvoice_psc.slurm` + the `common-voice/psc_array/` job arrays — see
**[psc-howto.md](psc-howto.md)** and `setup_psc.sh`.

Outputs (no corpus audio/features are committed — see *Data & licensing*):
- **TIMIT** → `timit/reports/report.md`, `timit/tables/`, `timit/reports/figs/`
- **Common Voice** → `$CV_ANALYSIS/reports/scaling_report.txt`, `.../tables/scaling_*.csv`, `.../reports/figs/scaling_all.png`

Two corpora, one shared corpus-agnostic analysis core:

## [`timit/`](timit/) — PR · summed-MI · Fano on TIMIT
The reference experiment on TIMIT (630 speakers, 6300 utterances, gold phone
alignments). Computes, pooled and within-sex:
- **PR** — participation ratio (effective dimensionality) of the between-speaker
  correlation matrix.
- **Summed MI** — debiased per-feature mutual information with speaker; an **upper**
  bound on joint speaker information.
- **Fano** — held-out speaker classifier with Fano / cross-entropy bounds; a
  **lower** bound.

Headline (pooled): PR ≈ 8 of 40, Fano lower bound ≈ 7.4 bits, ceiling log₂S = 9.30,
summed-MI ≈ 21 bits. See [`timit/reports/report.md`](timit/reports/report.md).

## [`common-voice/`](common-voice/) — speaker-count scaling on Common Voice
The Common Voice analogue plus a **speaker-count scaling sweep**: recompute the same
three quantities on nested, balanced speaker subsets of growing size and plot each
against N with the corpus ceiling log₂N. Reuses the TIMIT analysis core verbatim;
adds MFA-forced-alignment masking (no gold phones in CV) and the scaling driver.

Pilot (CV21 English, 200→1000 speakers): the **Fano lower bound tracks log₂N at a
~constant ~92% of ceiling** (slope 0.86) — speaker distinctiveness is corpus-limited,
not saturating. See [`common-voice/README.md`](common-voice/README.md).

## [`extractor/`](extractor/) — the 40-feature utterance extractor
The upstream feature extractor that produces the fixed 40-feature table both
analyses consume (F0/jitter/shimmer, glottal-source via IAIF, formants via a
hybrid DeepFormants + Burg estimator, harmonic/spectral features, VFI via DeepFry,
alignment-native speech-rate/VOT/BGD). Vendors **DeepFry** (Chernyak et al. 2022)
and **DeepFormants** (Dissen & Keshet 2016), both **MIT-licensed**, with provenance
and pinned commits/SHA-256s in each `third_party/*/VENDOR.md`. See
[`extractor/README.md`](extractor/README.md) and `extractor/details-of-extraction.md`.

## Shared core
Both experiments use the same corpus-agnostic modules (`common.py`, `pr.py`,
`mi.py`, `fano.py`, `vfp_hurdle.py`) — only the corpus, masking, and (for CV) the
scaling loop differ. Run `pytest` in either subdir.

## Data & licensing
**No corpus audio or per-utterance features are committed** — TIMIT is LDC-licensed;
Common Voice is CC0/CC-BY (its clip-ID manifest *is* committed). Each subdir's README
documents the exact rerun steps from a separately-obtained corpus copy.

Code: MIT — the analyses, the extractor wrapper, and the vendored DeepFry and
DeepFormants (both MIT; their `LICENSE`/`VENDOR.md` retained under
`extractor/third_party/`). The vendored model weights are redistributed here under
those MIT terms.
