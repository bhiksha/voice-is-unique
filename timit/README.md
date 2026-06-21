# timit — PR, summed-MI, and Fano on the TIMIT 40-feature speaker representation

Reviewer-reproducible analysis of how much **speaker identity** is carried by a fixed
40-feature, utterance-level acoustic representation of TIMIT (all 6300 utterances,
630 speakers). Three quantities, each in two conditions (**pooled** and **within-sex**):

- **PR** — effective dimensionality `d_eff` (participation ratio of the between-speaker
  correlation matrix).
- **Summed MI** — sum of debiased per-feature mutual informations with speaker; an
  **upper** bound on joint speaker information (over-counts redundancy).
- **Fano** — held-out speaker classifier with Fano / cross-entropy bounds; a **lower**
  bound on joint speaker information.

The bracket is `Fano ≤ joint info ≤ summed MI`, with corpus ceiling `H(Y) = log2 S`.

## Data availability

TIMIT is **LDC-licensed** (LDC93S1). This repository contains **only code, the report,
aggregate tables, and figures** — **no audio, no alignments, and no per-utterance feature
files**. Obtain TIMIT from the LDC, run the upstream feature extractor (the
`voice-is-unique` extractor) to produce `all_utterances.parquet`, then run the command
below. Every committed number is reproducible from your own corpus copy.

## Reproduce

```bash
pip install -r requirements.txt

# 1. Obtain TIMIT from the LDC and extract the 40-feature table (upstream extractor)
#    → produces ~/data/timit-feats/all_utterances.parquet
#    (one row per utterance: 40 features in fixed order + identifiers + per-feature
#     valid-frame counts; 0 vs NaN distinguishes state-2 from state-3, see CONFIG)

# 2. Point CONFIG/timit.json:input_parquet at that file (default ~/data/timit-feats/...)

# 3. Run everything (PR + MI + Fano, pooled + within-sex):
python -m src.run_all --config CONFIG/timit.json

# 4. Tests:
pytest -q
```

Outputs: `tables/` (aggregate CSVs + `provenance.json`), `reports/report.md`,
`reports/figs/`. Deterministic given the config and seeds (`seed: 0`).

## Headline results (pooled, TIMIT, input sha256 `812d61848620c064`)

- **PR ≈ 8** effective dimensions of 40 (Option C; robust to VFP modeling; null ≫ real).
- **Fano lower bound ≈ 7.3 bits** (cross-entropy, multinomial logistic), of a `9.30`-bit ceiling.
- **Summed-marginal MI** (Nbin=5) is the redundancy-inflated upper bound.

See `reports/report.md` for the full §0–§5 report (provenance, PR, MI, Fano, synthesis,
verification) and per-condition / within-sex numbers.

## Layout

```
CONFIG/timit.json     frozen config: transforms, VFP hurdle, bins, CV, seeds, floors
src/common.py         load / transform / z-score, data states, Fisher components, PR-from-cov
src/vfp_hurdle.py     VFP hurdle (presence r_i, magnitude M_i) + Option C reconstruction
src/pr.py             Σ_b (pairwise), PR, Option C/B/exclude, bootstrap, null, incremental
src/mi.py             quantile-bin MI, VFP zero-bin, weighted entropy, permutation debias
src/fano.py           speaker-stratified CV, train-only imputation, classifiers, bounds
src/run_all.py        one-command driver (pooled + within-sex) → tables + figures + report
src/report.py         report builder
tests/                acceptance tests (run with `pytest`)
```

## Method notes

- **No listwise row deletion.** Genuine NaNs (state 3) are **ignored** — never binned,
  indicated, or used as a cue: MI bins a feature's measured support only (no missing bin);
  PR uses pairwise-complete covariance; Fano mean-imputes state-3 cells (train-fold mean,
  **no indicator**) keeping all 6300 rows.
- **Three data states** per (utterance, feature): 1 measured `>0`; 2 measured `==0`
  (a real signal, e.g. VFP no-creak); 3 NaN (uncomputable). Kept distinct throughout.
- **VFP hurdle (Option C).** VFP's between-speaker covariance row is reconstructed via the
  law of total covariance from presence rate and log-magnitude, validated by a synthetic
  recovery test (`tests/test_vfp_hurdle.py`). Option B (presence-only) and exclude-VFP are
  reported as robustness. On TIMIT every speaker creaks (φ=1), so the speaker-level
  presence-variance term is zero.

## License

MIT (see `LICENSE`).
