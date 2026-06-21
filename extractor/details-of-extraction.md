# Details of feature extraction

How each of the 40 utterance-level features in `all_utterances.parquet` was
computed, with the exact parameters used. Everything below is pinned in
`src/timit_features/config.py` (a SHA-256 hash of that config is stamped into
every output record). Methodology decisions and their rationale are logged in
`DECISIONS.md`.

Corpus run: 6300/6300 TIMIT utterances, 630 speakers, 0 decode failures.

---

## 1. Common pipeline (every utterance)

1. **Decode** the NIST SPHERE `.WAV` with `soundfile` (libsndfile) to mono
   float64 at 16 kHz. A decode failure produces an all-NaN record (never raised).
   (`io_timit.py`)
2. **Parse alignments** `.PHN` (phones), `.WRD` (words), `.TXT` (transcript).
   Phone intervals are `start_sample end_sample label` at 16 kHz. (`io_timit.py`)
3. **Frame** on a fixed grid: window **25 ms** (400 samples), hop **10 ms**
   (160 samples); frame *i* spans `[i·hop, i·hop+win)`. (`framing.py`, `dsp.py`)
4. **Phone-class mask**: each frame is mapped to the phone whose `[start,end)`
   contains its **center sample**, then to a phone class, then to the per-feature
   **domains**. No energy VAD is ever used — masking is alignment-only. (`framing.py`)
5. **Per-feature extraction** produces a per-frame value array (NaN where
   undefined); alignment-native features are computed directly from `.PHN`.
6. **Aggregate** each frame feature over its domain mask ∩ finite values with its
   fixed statistic; if fewer than **MIN_VALID_FRAMES = 5** valid frames →
   the utterance value is NaN. (`aggregate.py`)

Pre-emphasis (coefficient **0.97**) is applied before the librosa/scipy spectral
and Burg-LPC analyses. Praat (F0/jitter/shimmer/CPP) manages its own pre-emphasis.

### Phone classes (TIMIT 61-phone set → class)
- **silence**: `h# pau epi`  (excluded everywhere)
- **vowel**: `iy ih eh ey ae aa aw ay ah ao oy ow uh uw ux er ax ix axr ax-h`
- **nasal**: `m n ng em en eng nx`
- **liquid**: `l el r`
- **glide**: `w y hh hv`
- **voiced_obstruent**: `b d g jh z zh v dh dx` **+ voiced closures `bcl dcl gcl`**
- **voiceless_obstruent**: `p t k ch s sh f th` **+ `q`**
- **closure (excluded)**: `pcl tcl kcl`

Voiced-stop closures carry the low-frequency *voice bar*, so they are treated as
voiced_obstruent (they enter voiced+speech domains, not formants); voiceless
closures are excluded like silence (used only to locate stop releases for VOT).

### Domains (which frames each feature uses)
- **voiced** = vowel + nasal + liquid + glide + voiced_obstruent
- **sonorant** = vowel + nasal + liquid + glide
- **speech** = all non-silence, non-closure
- **sibilant** = phones `s sh z zh` (SSPF only)

### Aggregation statistics
`median`, `mean`, `flux_mean` (mean of frame-to-frame flux), `sd_semitone`
(SD of `12·log2(F0/median F0)`), or `native` (alignment features).

---

## 2. Glottal source via IAIF (`iaif.py`, `features_glottal.py`)

The glottal flow is estimated **per voiced phone-segment** by **IAIF**
(Iterative Adaptive Inverse Filtering, Alku 1992), native numpy/scipy/librosa:
high-pass at **40 Hz**, 1st-order glottal pre-whitening, vocal-tract LPC of order
**18**, lip-radiation cancellation (leaky integrator coef **0.99**), then refined
with a glottal LPC of order **4** over **2 iterations**. Glottal closure instants
(GCIs) are the minima of the flow derivative (min spacing = `sr / F0_ceiling`).
Per glottal cycle `[GCI_k, GCI_{k+1})` the parameters below are read off the flow
and assigned to the frames the cycle covers; aggregated over **voiced** frames.

| # | Feature | Per-cycle computation | Agg |
|---|---|---|---|
| 1 | **F0** | (Praat — see §5) | median |
| 4 | **GCT** | closed-phase duration in ms (samples with flow < min+10%·range) | mean |
| 5 | **CQ** | closed-phase samples / cycle length (closed quotient) | mean |
| 6 | **MFDR** | max of −d(flow)/dt within the cycle (peak flow declination) | mean |
| 7 | **SQ** | opening-phase / closing-phase duration (flow min→max vs max→end) | mean |
| 8 | **NAQ** | `f_AC / (d_peak · T0)` (Alku et al. 2002): peak-to-peak flow ÷ (MFDR · period) | mean |

*Caveat:* CQ/SQ/GCT closed-phase geometry uses a 10%-of-range floor threshold —
an operational choice flagged for calibration in DECISIONS.

---

## 3. Formants — hybrid (`features_formant.py`)

- **F1–F4** from **DeepFormants** (Chernyak/Dissen/Keshet; vendored PyTorch model,
  arXiv-era), run in the isolated `deepformants` conda env as a subprocess on a
  temporary PCM WAV (so SPHERE never reaches it). One model-load per utterance;
  DeepFormants is run **per sonorant phone-segment** and the F1–F4 estimate is
  assigned to that segment's frames.
- **F5 and all bandwidths B1–B5** from **order-20 Burg LPC** computed per sonorant
  frame on the pre-emphasised signal at **native 16 kHz** (no downsampling). LPC
  roots → pole frequencies/bandwidths.
- **Pole matching (spurious-peak rejection):** each Burg pole is matched to the
  DeepFormants F1–F4 within **300 Hz**; the matched poles give **B1–B4**. **F5** is
  the lowest unmatched pole above F4, below the sex ceiling, with bandwidth
  < **1000 Hz**; its bandwidth is **B5**.
- Sex-dependent formant ceiling: **5000 Hz (male) / 5500 Hz (female)** (sex from
  the TIMIT speaker-dir initial). Aggregated as **median over sonorant frames**.

Features #13–17 = F1–F5; #18–22 = B1–B5.

---

## 4. Resonance — Nasality (#23) (`features_nasality.py`)

**A1 − P0** (Chen 1997), a single-channel acoustic nasalization correlate (no
Nasometer, no model). Per **sonorant** frame from the pre-emphasised power
spectrum: **A1** = peak amplitude (dB) in the window `F1·(1 ± 0.2)`; **P0** = peak
amplitude (dB) in the nasal-murmur band **200–500 Hz**; Nasality = **A1 − P0**
(lower ⇒ more nasal). Mean over sonorant frames. (F1 from §3.)

---

## 5. Praat measures (`features_praat.py`, via parselmouth)

- **F0 (#1)** — Praat **autocorrelation** (`To Pitch (ac)`), 10 ms step,
  sex-dependent range (M **75–300 Hz**, F **100–500 Hz**), Praat default voicing
  parameters (silence 0.03, voicing 0.45, octave-cost 0.01, octave-jump 0.35,
  voiced/unvoiced 0.14, max 15 candidates). Per-frame contour sampled at frame
  centers (nearest Praat frame). **median over voiced frames.**
- **semitone_SD_F0 (#12)** — SD of `12·log2(F0/median F0)` over voiced frames.
- **jitter (#2), shimmer (#3)** — computed from a **cross-correlation
  PointProcess** (Praat Voice Report convention: local jitter; local shimmer;
  period bounds 0.1–20 ms, max period factor 1.3, max amplitude factor 1.6),
  **per voiced phone-segment**, assigned to that segment's frames. **mean.**
- **CPP (#34)** — Praat **PowerCepstrogram CPPS** (smoothed), per voiced segment.
  **mean over voiced frames.**

*Caveat:* jitter/shimmer/CPP use per-segment Praat estimates (they need several
periods), so the frame-mean weights by segment duration.

---

## 6. Harmonic / voice-quality (`features_harmonic.py`)

Computed from the per-frame power spectrum and the F0/CPP contours.

| # | Feature | Computation | Domain/Agg |
|---|---|---|---|
| 9 | **SHR** | subharmonic/harmonic amplitude ratio: Σ amp at `(k−½)·F0` ÷ Σ amp at `k·F0` (Sun-style) | voiced, mean |
| 10 | **IHI** | inharmonicity: mean \|peak_freq − k·F0\| over harmonics up to 5 kHz (Hz) | voiced, mean |
| 11 | **VFI** | vocal-fry index — see §7 (DeepFry) | voiced, mean |
| 32 | **GNE** | Glottal-to-Noise Excitation (Michaelis-style): max cross-correlation of Hilbert envelopes of LPC-residual band-pass signals across bands in **0.3–4 kHz** (bandwidth 1 kHz, hop 300 Hz, pairs ≥½-bandwidth apart) | voiced, mean |
| 35 | **dCPP** | frame-wise \|ΔCPP\| (absolute change in CPP between consecutive frames) | voiced, mean |
| 37 | **AMD** | amplitude-modulation depth: coefficient of variation of the **<20 Hz** low-passed RMS envelope over speech frames | speech, mean |

*Caveat:* IHI/SHR are simplified relative to canonical references; AMD uses a CoV
proxy — flagged in DECISIONS.

---

## 7. VFI (#11) via DeepFry (`deepfry_creak.py`)

Creak is detected by **DeepFry** (Chernyak et al., Interspeech 2022; arXiv
2203.17019; paper model `CREAK-220`), run in the isolated `deepfry` conda env
(Python 3.8 / torch 1.12 CPU) as a subprocess in `--custom` mode on a temp PCM
WAV. The predicted creak intervals are read from the output TextGrid `pred-creaky`
tier (marks `"c"`). Each **voiced** frame gets a 0/1 creak indicator (1 if its
center falls in a creak interval); the **mean over voiced frames** is the Vocal
Fry Index (= proportion of voiced speech in creak).

**Zero is a valid value, not "missing".** A voiced utterance in which no creak is
detected has VFI = 0 (no fry), and VFI is **NaN only when the utterance has no
voiced frames at all**. This is enforced in two places: `deepfry_creak.compute`
sets every voiced frame to 0 up front (so a DeepFry failure or empty result still
yields 0, never NaN), and `extract.py` applies a final guard — if VFI is NaN but
the utterance has ≥1 voiced frame (e.g. the min-valid-frames guard fired, or
DeepFry crashed), it is forced to 0. DeepFry has one retry to absorb transient
parallel-run failures. Consequently VFI is analysed **raw** (no log transform):
an earlier `log_nonzero` mapping (0 → NaN → log) wrongly discarded the ~489
zero-fry utterances and is no longer used. Over the corpus: 0 NaN, 482 zeros,
5818 positive, range [0, 0.76].

---

## 8. Spectral & energy (`features_spectral.py`, `dsp.py`)

Per-frame from the pre-emphasised, Hann-windowed power spectrum (rfft of the
400-sample frame). Aggregated as **mean over speech frames** (flux as flux_mean).

| # | Feature | Computation |
|---|---|---|
| 24 | **spectral_skewness** | 3rd standardized moment of the power-vs-frequency distribution |
| 25 | **spectral_kurtosis** | 4th standardized moment |
| 26 | **spectral_entropy** | Shannon entropy of the normalized power spectrum (nats) |
| 27 | **spectral_rolloff** | frequency below which **85 %** of energy lies |
| 28 | **spectral_flux** | Euclidean change of the magnitude spectrum vs the previous frame (mean) |
| 29 | **alpha_ratio** | energy(1–5 kHz) / energy(50–1000 Hz) (linear band-energy ratio) |
| 30 | **LHR** | energy(<1 kHz) / energy(>3 kHz) (linear) |
| 31 | **SPI** | energy(≥4 kHz) / energy(<4 kHz) — Hillenbrand & Houde (1996) "H/L" |
| 33 | **SSPF** | sibilant spectral peak: argmax-frequency above 2 kHz, on **sibilant** frames |
| 36 | **RMS** | per-frame RMS of the raw (non-pre-emphasised) signal |

*Caveat:* alpha_ratio/LHR/SPI are **linear** band-energy ratios (could be dB);
their distributions are right-skewed.

---

## 9. Alignment-native timing/prosody (`features_alignment.py`)

Computed directly from `.PHN`; **event-based validity** (NaN iff no qualifying
event), not the frame guard.

| # | Feature | Computation |
|---|---|---|
| 38 | **speech_rate** | vowel-class nuclei ÷ speaking span (first→last non-silence) = syllables/sec |
| 39 | **VOT** | mean release-segment duration over closure→release stop pairs (`Xcl`→`X`, X∈b d g p t k); NaN if no release. *(Sign of prevoiced stops not modeled.)* |
| 40 | **BGD** | mean duration of maximal non-silence runs between `h#`/`pau`/`epi` |

---

## 10. Fixed feature order (columns 1–40)
F0, jitter, shimmer, GCT, CQ, MFDR, SQ, NAQ, SHR, IHI, VFI, semitone_SD_F0,
F1, F2, F3, F4, F5, B1, B2, B3, B4, B5, Nasality, spectral_skewness,
spectral_kurtosis, spectral_entropy, spectral_rolloff, spectral_flux, alpha_ratio,
LHR, SPI, GNE, SSPF, CPP, dCPP, RMS, AMD, speech_rate, VOT, BGD.

Each `all_utterances` row also carries `cov_<feature>` (valid frames / events that
entered the aggregation) and identifiers (basename, rel_path, speaker_id, sex,
dialect_region, split, sample_rate, duration, decode_ok, config_hash).

---

## 11. Determinism & reproducibility
All estimators run CPU-only with fixed seeds; the pipeline is deterministic
(verified byte-identical on re-runs). Results are independent of `--jobs`. The
two external models are vendored at pinned commits with recorded SHA-256 model
hashes (`third_party/deepformants`, `third_party/deepfry`). The exact parameter
set is in `CONFIG.json` and hashed into every record.
