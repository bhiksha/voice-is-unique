# PR · summed-MI · Fano on the TIMIT 40-feature speaker representation

Three quantities on the fixed 40-feature utterance-level representation, all 6300 utterances, pooled and within-sex. Summed MI is an UPPER bound and Fano a LOWER bound on joint speaker information; the corpus ceiling is log2 S.

## §0 Provenance & freeze
- input `/home/bhiksha/data/timit-feats/all_utterances.parquet` sha256 `812d61848620c064`
- **S = 630 speakers, N = 6300 utterances**, balanced (n_i = 10 for every speaker) ⇒ H(Y) = log2 S = 9.2992 bits
- per sex: S_F = 192 (log2 = 7.5850), S_M = 438 (log2 = 8.7748)
- transforms: NAQ, alpha_ratio, LHR, SPI → natural log; **VFP (=VFI) → hurdle** (log of non-zero state-1 magnitude, z-scored over state-1; exact zeros = state-2 carried by the hurdle: MI zero-bin, PR Option C, Fano presence). All others linear; then corpus z-score to unit variance.
- SIGMA_B_FLOOR=0.001, PSD-repair=nearest_psd_higham, N_BOOT=1000; MI Nbin∈[2, 3, 5, 10] (ref 5), n_perm=200; Fano 5-fold speaker-stratified, classifiers=['logreg', 'lda', 'mlp'], impute=train-fold-mean (no indicator); seed=0.
- libs: python 3.11.15, numpy 2.4.6, pandas 3.0.3; 2026-06-18T21:28:26.199108+00:00

## §1 PR — effective dimensionality (participation ratio)
- **pooled:** PR Option C **8.074** (95% CI [7.671, 8.386]) · Option B 8.231 · exclude-VFP 7.992 · null 11.849±0.317 · 90%-var dim 18 · spec-entropy dim 15.09
- **within F:** PR Option C **13.064** (95% CI [11.532, 13.147]) · Option B 13.211 · exclude-VFP 12.968 · null 15.948±0.427 · 90%-var dim 19 · spec-entropy dim 19.23
- **within M:** PR Option C **14.026** (95% CI [13.070, 14.242]) · Option B 14.441 · exclude-VFP 14.207 · null 16.817±0.368 · 90%-var dim 20 · spec-entropy dim 20.46
- **within-sex combined** Σ_g (S_g/S)·PR_g: Option C 13.733 (exclude-VFP 13.830)
- VFP contribution to PR (pooled): exclude 7.992 → Option C 8.074; Option C vs B gap 0.158 bounds the reconstruction's influence. Null ≥ real in every case ⇒ features are genuinely redundant.
- φ = 1 on TIMIT (every speaker creaks in ≥1 utterance), so Option C's presence-variance term is exactly 0; VFP's between-speaker signal is carried by the rate×magnitude composite.

## §2 Summed MI — per-feature, debiased (UPPER bound)
Top features by I_corrected (pooled, Nbin=5):
| feature | n_used | I_raw | I_null | I_corrected |
|---|---|---|---|---|
| F0 | 6300 | 1.7366 | 0.3373 | 1.3993 |
| SHR | 6300 | 1.6066 | 0.3371 | 1.2695 |
| IHI | 6300 | 1.4346 | 0.3364 | 1.0982 |
| F4 | 6300 | 1.3998 | 0.3378 | 1.0620 |
| MFDR | 6300 | 1.2553 | 0.3371 | 0.9182 |
| F5 | 6300 | 1.2323 | 0.3379 | 0.8945 |
| RMS | 6300 | 1.2177 | 0.3369 | 0.8808 |
| GCT | 6300 | 1.1702 | 0.3365 | 0.8337 |
| F3 | 6300 | 1.1392 | 0.3373 | 0.8020 |
| spectral_flux | 6300 | 1.1275 | 0.3371 | 0.7904 |

- **summed-marginal MI (UPPER bound)** Σ_f I_corrected, pooled: Nbin=2: 13.80 bits, Nbin=3: 18.81 bits, Nbin=5: 20.99 bits, Nbin=10: 18.94 bits
- within-sex combined summed MI (Nbin=5): 16.46 bits
- VFP zero-bin (presence/absence) contributes I_corrected=0.0545 of VFP total 0.4484 — most VFP info is magnitude, not presence.
- Genuine NaNs (state 3) are IGNORED: no missing bin; each feature's MI uses only its measured support (full per-feature table in `tables/mi_table.csv`).

## §3 Fano — joint-information LOWER bound (all 6300 utterances)
| condition | classifier | acc | I_fano | I_xent |
|---|---|---|---|---|
| pooled | logreg | 0.731±0.005 | 5.96±0.06 | 7.32±0.03 |
| pooled | lda | 0.783±0.010 | 6.53±0.11 | 7.44±0.11 |
| pooled | mlp | 0.700±0.013 | 5.63±0.13 | 7.39±0.09 |
| within_F | logreg | 0.763±0.019 | 5.00±0.18 | 5.93±0.07 |
| within_F | lda | 0.791±0.015 | 5.26±0.14 | 5.82±0.10 |
| within_F | mlp | 0.721±0.018 | 4.62±0.16 | 5.98±0.08 |
| within_M | logreg | 0.762±0.016 | 5.90±0.17 | 7.03±0.05 |
| within_M | lda | 0.803±0.012 | 6.33±0.13 | 7.11±0.15 |
| within_M | mlp | 0.741±0.016 | 5.68±0.17 | 7.20±0.05 |

- **pooled headline (max over classifiers):** I_fano 6.53, I_xent 7.44 bits (ceiling 9.30).
- within-sex combined headline: I_fano 6.00, I_xent 6.83 bits.
- permutation null (pooled, shuffled Y): acc 0.00079 ≈ 1/S, I_fano 0.00.
- state-3 mean-imputation counts (no indicator): SSPF=365, VOT=238; all others 0. N=6300 throughout.
- Capacity inversion (MLP ≤ linear) is a data-starvation diagnostic (8 train utts/speaker).

## §4 Synthesis — the bracket
- **pooled:** joint speaker info ∈ [Fano lower **7.44**, ceiling **9.30**] bits; summed-marginal MI 21.0 bits is the redundancy-inflated UPPER bound; PR ≈ **8.1** effective dimensions of 40.
- derived per-axis resolution q_eff = 2^(Fano/PR) = 1.89 (consistency check; derived, not measured).
- pooled vs within-sex (the pooled→within gap = sex-parent contribution): PR 8.07→13.73, Fano 7.44→6.83, summed MI 21.0→16.5.

## §5 Verification
- nulls: PR null 11.85 ≥ real 8.07; MI debiased by 200-shuffle null; Fano null acc≈1/S, bounds≈0.
- Option C validated by the synthetic-recovery acceptance test (`tests/test_vfp_hurdle.py`).
- determinism: fixed seeds; same config + input → identical outputs.
- caveats: single-session TIMIT ⇒ optimistic (no cross-session within-speaker variation); PR is linear/2nd-order; bounds are corpus-limited (S=630).

Figures: `reports/figs/`. Aggregate tables: `tables/`.
