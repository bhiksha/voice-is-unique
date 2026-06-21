# voice-is-unique

How much **speaker identity** is carried by a fixed 40-feature, utterance-level
acoustic representation of voice — measured three ways (effective dimensionality,
an upper bound, and a lower bound on joint speaker information) and shown to scale
with the number of speakers rather than saturating at a voice-imposed limit.

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

## Shared core
Both experiments use the same corpus-agnostic modules (`common.py`, `pr.py`,
`mi.py`, `fano.py`, `vfp_hurdle.py`) — only the corpus, masking, and (for CV) the
scaling loop differ. Run `pytest` in either subdir.

## Data & licensing
**No corpus audio or per-utterance features are committed** — TIMIT is LDC-licensed;
Common Voice is CC0/CC-BY (its clip-ID manifest *is* committed). Each subdir's README
documents the exact rerun steps from a separately-obtained corpus copy. The upstream
40-feature extractor (DeepFry/DeepFormants-based) is a separate component.

Code: MIT (see each subdir's `LICENSE`).
