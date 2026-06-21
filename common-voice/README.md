# common-voice — speaker-count scaling of PR, summed-MI, and Fano

The **Common Voice** analogue of the TIMIT PR/MI/Fano experiment, plus a **speaker-count
scaling sweep**. It tests whether recoverable speaker information and effective
dimensionality **scale with the number of speakers** (tracking `log2 N`) rather than
saturating at a voice-imposed plateau. Reuses the corpus-agnostic TIMIT analysis core
(`src/common.py`, `pr.py`, `mi.py`, `fano.py`, `vfp_hurdle.py`) verbatim.

## What scales

For nested, balanced speaker subsets of growing size N (1000 → 10000), recompute **PR**,
**summed-MI** (upper bound), and **Fano** (lower bound), pooled and within-sex, and plot
each vs N with the ceiling `log2 N`. Slopes ≈ 1 vs `log2 N` ⇒ the **corpus**, not the
voice, is the binding constraint.

## Configuration (frozen in `CONFIG/common_voice.json`)

- **Release:** Common Voice **21.0**, English.
- **Selection:** speakers with ≥ 100 clips of ≥ 5.0 s; **5000 male + 5000 female**
  (binary self-reported gender only — missing/other/non-binary excluded); exactly 100
  clips/speaker (first by sorted clip id); seeded.
- **Masking:** **MFA** forced alignment (`english_us_arpa`) → TIMIT-comparable phone-class
  masks (no gold alignments in CV — a frozen new degree of freedom; cross-corpus
  comparison to TIMIT carries this caveat).
- **Transforms / states / hurdle:** identical to TIMIT (NAQ/alpha_ratio/LHR/SPI = natural
  log; VFP = log-nonzero hurdle; NaNs ignored). Transform + z-score statistics computed
  on the **full 10,000-speaker set and reused at every subset** (only N varies).
- **Scaling:** per-sex sizes 500…5000 (pilot 100/250/500); **fixed 100 clips/speaker** at
  every size; **fixed full-set standardization**.

Multi-session CV ⇒ realistic within-speaker variance ⇒ **honest (non-inflated)** F*/MI/Fano.

## Reproduce (end to end, from nothing but this repo + the public CV release)

```bash
pip install -r requirements.txt
# forced aligner:
conda create -n aligner -c conda-forge montreal-forced-aligner
conda run -n aligner mfa model download acoustic english_us_arpa
conda run -n aligner mfa model download dictionary english_us_arpa

# 1. Accept Common Voice 21.0 terms on Hugging Face and log in:
huggingface-cli login                       # gated dataset — your account must accept terms

# 2. select speakers + clips → manifest (committed):
python -m src.download --config CONFIG/common_voice.json --metadata-tsv <cv21_validated.tsv>

# 3. extract 40 features (MFA-masked) → ~/data/commonvoice-feats/all_utterances.parquet:
python -m src.extract --config CONFIG/common_voice.json

# 4. scaling sweep + headline figure (use --pilot for 100/250/500):
python -m src.run_all --config CONFIG/common_voice.json [--pilot]
pytest -q
```

Outputs: `tables/scaling_*.csv`, `reports/figs/scaling_*.png`, `reports/report.md`.
Audio and per-clip features go to `~/data/commonvoice` and `~/data/commonvoice-feats`
(gitignored). **Only code, aggregate results, figures, and the clip-ID manifest are
committed** — CV is CC0/CC-BY, so the manifest may be shared (unlike TIMIT).

## Data availability

Common Voice is CC0/CC-BY (`mozilla-foundation/common_voice_21_0`, gated by terms
acceptance). The committed `manifest/selected_clips.csv` lets a reviewer reconstruct the
exact 10,000-speaker × 100-clip set; `src/download.py` + `src/extract.py` make the
pipeline reproducible from the public release.

## Layout

```
CONFIG/common_voice.json   frozen config (release, selection, MFA, transforms, scaling grid)
src/common,pr,mi,fano,vfp_hurdle.py   reused TIMIT analysis core (corpus-agnostic)
src/download.py            CV selection + clip-ID manifest (seeded)
src/extract.py             MFA masking + 40-feature extraction → parquet
src/scaling.py             nested-subset sweep (fixed clips, fixed standardization)
src/run_all.py             end-to-end driver (select → extract → analyze)
tests/                     reused TIMIT tests + scaling invariants (nested/fixed-clips/fixed-std)
```

## License

MIT (see `LICENSE`).
