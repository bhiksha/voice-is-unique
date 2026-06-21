# Open methodology decisions — needs your sign-off before implementation

Per the prompt's hard rules, no parameter that affects a number is finalized
until you approve it, and features whose *definition* is uncertain will not be
implemented until you confirm the definition/reference. This file is the single
list of every such decision. Answer inline (edit the **→ DECISION:** lines) or
in chat. Items are grouped; `config.py` carries the matching `❓CONFIRM` markers.

Legend: **[param]** = numeric/parameter choice · **[def]** = feature definition.

---

## A. Output location
1. **[param]** The prompt says write `timit-feats/` as a *sibling of the TIMIT
   root*. The TIMIT root is `~/data/timit/TIMIT`, so output would be
   `~/data/timit/timit-feats/`. That writes into the corpus's parent. OK, or
   would you prefer it under the project (`~/claude/voice-is-unique/timit-feats/`)?
   → DECISION: **`~/data/timit-feats/`** (decoupled from the corpus tree;
   `config.DEFAULT_OUTPUT_ROOT`, CLI `--out` may override). [ANSWERED 2026-06-15]

## B. Framing & pre-processing (`FramingConfig`)
2. **[param]** Frame length / hop: proposed **25 ms / 10 ms**. Confirm.
   → DECISION: **25 ms / 10 ms**. [ANSWERED 2026-06-15]
3. **[param]** Analysis window for the librosa/scipy spectral features:
   proposed **Hann**. Confirm.
   → DECISION: **Hann**. [ANSWERED 2026-06-15]
4. **[param]** Pre-emphasis coefficient before spectral/LPC features: proposed
   **0.97**. Confirm, or none.
   → DECISION: **0.97**. [ANSWERED 2026-06-15]

## C. F0 analysis (`PitchConfig`)
5. **[param]** F0 range: **single fixed 75–500 Hz** for all speakers, OR
   **sex-dependent** (male 75–300, female 100–500)? Sex is known from the TIMIT
   dir name, so either is available. This affects octave errors and therefore
   F0, jitter, shimmer, CQ, NAQ, semitone_SD_F0, SHR, CPP, GNE.
   → DECISION: **sex-dependent** (sex from speaker-dir initial M/F).
   [ANSWERED 2026-06-15]
5b. **[param]** Exact F0 bounds.
    → DECISION: male **75–300 Hz**, female **100–500 Hz**. [ANSWERED 2026-06-15]
5c. **[param]** F0 extraction **method** + voicing-decision params (parselmouth/Praat).
    Available: ac (autocorrelation), cc (cross-correlation), shs, spinet.
    Proposal: **`ac`** (Boersma 1993, Praat's recommended F0/intonation method) for
    F0 (#1) and semitone_SD_F0 (#12); time step = 10 ms (= hop). Voicing-decision
    params at Praat ac defaults (silence 0.03, voicing 0.45, octave_cost 0.01,
    octave_jump 0.35, voiced_unvoiced 0.14, max_candidates 15) — these set which
    frames count as voiced, hence Group-2 valid-frame counts. Jitter/shimmer (#2/#3)
    computed from a **cc**-derived PointProcess (Praat Voice Report convention).
    → DECISION: **Praat autocorrelation (ac)**, 10 ms step, Praat-default voicing
    params; jitter/shimmer from a **cc** PointProcess. Chosen over pYIN after an
    empirical comparison (16 TIMIT utts, both sexes): median-F0 diff ~8.5 cents,
    voiced yield 77.4% vs 78.3%, octave disagreement 0.5% — a wash, and Praat-ac
    keeps one estimator consistent with the Praat-based voice-quality features.
    [ANSWERED 2026-06-15]

## D. Formant analysis (`FormantConfig`)
6. **[param]** Formant ceiling: **single 5500 Hz**, OR **sex-dependent**
   (5000 male / 5500 female, Praat's usual convention)?
   → DECISION: **sex-dependent** — male **5000 Hz**, female **5500 Hz**.
   [ANSWERED 2026-06-15]
7. **[param]** Praat formant method = Burg, max 5 formants, window 25 ms.
   Confirm (or specify LPC order explicitly instead).
   → DECISION: **SUPERSEDED by #D (hybrid estimator).** [ANSWERED 2026-06-15]

### D. Formant estimator — HYBRID (DeepFormants + Burg LPC)
Searched for a maintained DL formant tool giving F5+bandwidths: none exists
(DeepFormants is Lua-Torch, F1–F4 only, no bandwidths). Precedent for the hybrid:
Gowda/Kadiri/Alku, Computer Speech & Language 2023 (arXiv 2308.09051).
→ DECISION [ANSWERED 2026-06-15]:
  - **F1–F4 frequencies** from **DeepFormants** (trained on VTR-TIMIT).
  - **F5 + B1–B5** from **order-20 Burg LPC at native 16 kHz** (NO downsampling —
    16 kHz supports the higher order; this also closes the earlier resample question).
  - **Spurious-pole rejection:** match Burg poles to DeepFormants F1–F4; B1–B4 are
    the bandwidths of the matched poles; F5 = lowest-frequency Burg pole above the
    matched F4, below the sex ceiling, with bandwidth < `f5_max_bandwidth_hz`; B5 is
    its bandwidth. No qualifying pole → F5,B5 = NaN.
  - Sex-dependent ceiling 5000/5500 Hz retained as the **max accepted formant freq**.

7b. **[param]** Burg analysis window: proposed **25 ms** (= frame length). Confirm.
    → DECISION: **25 ms**. [ANSWERED 2026-06-15]
Da. **[param]** Pole↔DeepFormants match tolerance `df_match_tolerance_hz`:
    proposed **300 Hz**. Confirm.
    → DECISION: **300 Hz**. [ANSWERED 2026-06-15]
Db. **[param]** Max bandwidth for a valid F5 pole `f5_max_bandwidth_hz`: proposed
    **1000 Hz**. Confirm.
    → DECISION: **1000 Hz**. [ANSWERED 2026-06-15]
Dc. **[infra]** DeepFormants is an unmaintained Lua-Torch / Python-2 stack. Running
    it deterministically in 2026 needs a feasibility spike (likely a pinned
    container or a port). If it can't be made reproducible, fall back to Burg-only
    for all of F1–F5 (plan C). Do the spike now, or after pinning #8–35?
    → SPIKE DONE 2026-06-15: **FEASIBLE via the repo's bundled PyTorch path** —
    NO Lua Torch, NO scikits.talkbox needed.
      • Estimator (MLP 350→1024→512→256→4) and Tracker (LSTM, 350→512→256→4) ship
        as `.pt` files in `pytorchFormants/`. Pure-Python LPC (`levinson_lpc.py`)
        replaces talkbox.
      • Ran the estimator end-to-end on the shipped example: **byte-identical
        across two runs** (deterministic), feature dim 350, plausible F1–F4
        (458/1643/2510/3689 Hz; shipped Lua reference was 508/1605/2671/3639 —
        small offset expected, pytorch vs original Lua model).
      • Needed exactly two patches for numpy-2.4/scipy-1.17: `np.fromstring`→
        `np.frombuffer`, `scipy.signal.hamming`→`scipy.signal.windows.hamming`.
      • Repro pinning recorded: repo commit `53e2541`; model sha256
        Estimator `1be6ba5a…`, Tracker `b799e1aa…`.
    Open follow-ups (Dd, De below).

Dd. **[infra]** Torch ABI. `pip install torch` into the analysis env caused a
    GLIBCXX clash for the *bare* interpreter (works fine under conda activation,
    which is how the pipeline runs). Proposal: **isolate DeepFormants in its own
    pinned conda env** (`deepformants`, conda-forge `pytorch-cpu`) invoked as a
    subprocess, keeping the `voice-is-unique` analysis env torch-free and clean.
    → DECISION: **Yes — isolate in its own `deepformants` env** (conda-forge
    pytorch-cpu), invoked as a subprocess. [ANSWERED 2026-06-15]
De. **[infra]** Vendor a patched DeepFormants (pin commit `53e2541` + the two
    patches + a wandb-free inference wrapper for the tracker) inside the project,
    or keep it external and document setup?
    → DECISION: **Yes — vendor it** under `third_party/deepformants/`.
    [ANSWERED 2026-06-15]

## E. Validity guard (`AggregationConfig`)
8. **[param]** `MIN_VALID_FRAMES` (a feature with fewer valid in-domain frames
   in an utterance → NaN): proposed **5**. Confirm.
   SCOPE (clarified): applies ONLY to frame-level features (#1-37), counting valid
   in-domain frames summed over the WHOLE utterance — NOT per segment/event. The
   alignment-native features use event-based validity instead, NOT a frame count:
     - VOT (#39)         → NaN iff no identifiable stop release (≥1 required).
     - speech_rate (#38) → NaN iff zero non-silence phones / zero speech duration.
     - BGD (#40)         → NaN iff zero speech runs.
   → DECISION: **MIN_VALID_FRAMES = 5** for frame-level features (#1-37).
   [ANSWERED 2026-06-15]. Native-feature validity rules (VOT/speech_rate/BGD,
   as above) to be finalized with their definitions in #38-40.

## F. Phone → class mapping (`PHONE_CLASS`) — the masking that drives every domain
9. **[param]** Stop **closures** (`bcl dcl gcl pcl tcl kcl`): proposed to treat
   as `closure` and **exclude from all acoustic features** (they are near-silent),
   while still using them to locate stop releases for VOT. Confirm.
   → DECISION: **Split (option 1).** Voiceless `pcl/tcl/kcl` → `closure`
   (excluded; VOT only). Voiced `bcl/dcl/gcl` → `voiced_obstruent` (voice bar →
   in voiced + speech domains incl. jitter/shimmer; excluded from formants since
   not sonorant). [ANSWERED 2026-06-15]
10. **[param]** `hh` (voiceless /h/) and `hv` (voiced /h/): proposed **glide**.
    Alternative: `hh`→voiceless_obstruent, `hv`→glide. Confirm.
    → DECISION: **both → glide.** [ANSWERED 2026-06-15]
11. **[param]** `dx` (flap): proposed **voiced_obstruent**. Alt: liquid / other.
    → DECISION: **voiced_obstruent.** [ANSWERED 2026-06-15]
12. **[param]** `q` (glottal stop): proposed **other** (excluded from all
    domains). Alt: treat as silence, or as voiceless_obstruent. Confirm.
    → DECISION: **voiceless_obstruent** (→ enters the speech domain; not voiced,
    not sonorant). [ANSWERED 2026-06-15]
13. **[param]** Domain set memberships (confirm all):
    - **voiced** = vowel + nasal + liquid + glide + voiced_obstruent
    - **sonorant** = vowel + nasal + liquid + glide
    - **speech** = all non-silence, non-closure
    → DECISION: **confirmed as listed** (silence/closure excluded everywhere).
    [ANSWERED 2026-06-15]

## G. "Standard" features — small parameter confirmations
14. **[param]** `spectral_rolloff` percentage: proposed **85%** (librosa default).
    → DECISION: **85%**. [ANSWERED 2026-06-15]
15. **[param]** Spectral-moments domain: §3 says "state whether voiced-only or
    all-speech." Proposed **all-speech** (matches §4 "mean over speech frames").
    Confirm.
    → DECISION: **all-speech**. [ANSWERED 2026-06-15]
16. **[def]** `CPP` (#34): cepstral peak prominence — proposed Praat
    `PowerCepstrogram` CPPS (Hillenbrand-style, smoothed). Confirm this is the
    intended CPP (vs unsmoothed CPP).
    → DECISION: **Praat PowerCepstrogram CPPS (smoothed)**. [ANSWERED 2026-06-15]

## H. Features whose DEFINITION must be confirmed before I implement them
For each, I will NOT write code until you confirm the definition/reference; a
wrong implementation is worse than NaN (your §4). My proposal is listed; replace
or confirm.

17. **[def] GCT (#4)** — not listed in the §3 group text (which starts at CQ), so
    its meaning is unclear to me. Glottal Closure Time? Glottal Cycle ...? 
    → DECISION: **GCT = Glottal Closure Time** = absolute duration of the closed
    phase per glottal cycle, mean over voiced frames, units **ms**. Distinct from
    CQ (#5, the closed/period *ratio*). [NAME ANSWERED 2026-06-15]
    Front-end = **IAIF** (per #18). GCT = closed-phase duration read off the IAIF
    glottal flow, mean over voiced frames, ms. [METHOD ANSWERED 2026-06-15]
    IAIF sub-params pending (see #18 block).
18. **[def] CQ (#5)** — Closed Quotient. No EGG in TIMIT, so this needs glottal
    inverse filtering (e.g. IAIF) + closed-phase detection. Confirm method/toolkit
    (IAIF? Covarep? aparat?), or drop to NaN.
    → DECISION: **IAIF** (Alku 1992) glottal inverse filtering; CQ = closed/period
    from the IAIF flow. IAIF is now the **shared glottal-source front-end** for
    GCT/CQ/NAQ/MFDR/SQ. [METHOD ANSWERED 2026-06-15]
    OPEN — IAIF sub-params (config `GlottalSourceConfig`), affect all five:
      vt_lpc_order 18, glottal_lpc_order 4, n_iterations 2, lip_radiation 0.99,
      highpass 40 Hz, GCI/GOI from flow derivative. Confirm or override.
    INFRA: implement IAIF natively in numpy/scipy (compact, deterministic, Alku
    1992 ref) in the torch-free analysis env — no extra heavy dependency.
19. **[def] MFDR (#6)** — Maximum Flow Declination Rate, from the glottal flow
    derivative (needs inverse filtering). Confirm method/reference.
    → DECISION: **MFDR = peak negative amplitude of the IAIF glottal-flow
    derivative within each cycle (|min of dU/dt|, at the GCI); mean over voiced
    frames.** [ANSWERED 2026-06-15]
20. **[def] SQ (#7)** — Speed Quotient (glottal opening/closing time ratio).
    Needs glottal flow. Confirm method/reference.
    → DECISION: **SQ = opening-phase / closing-phase duration of the IAIF flow per
    cycle (opening = GOI→flow peak; closing = flow peak→GCI); mean over voiced
    frames.** [ANSWERED 2026-06-15]
21. **[def] NAQ (#8)** — Normalized Amplitude Quotient (Alku et al. 2002). Needs
    IAIF glottal inverse filtering. Confirm toolkit/reference.
    → DECISION: **NAQ = f_AC / (d_peak · T0)** (Alku, Bäckström & Vilkman 2002),
    where f_AC = peak-to-peak AC flow amplitude, d_peak = |min of flow derivative|
    (= MFDR), T0 = period; from the IAIF flow, mean over voiced frames.
    [ANSWERED 2026-06-15]
22. **[def] SHR (#9)** — Subharmonic-to-Harmonic Ratio. Proposed: Sun (2002) SHRP
    algorithm. Confirm.
    → DECISION: **Sun (2002) SHRP** — subharmonic/harmonic amplitude ratio from the
    log-frequency spectrum (NOT IAIF-based); mean over voiced frames. Implement by
    porting shrp.m or a vetted Python port. [ANSWERED 2026-06-15]
### Definitions adopted from the paper (arXiv 2506.18182, Table 1). [ref] = paper's citation.
23. **IHI (#10)** → **Inharmonicity Index**: mean deviation of harmonic
    frequencies from integer multiples of F0 [44]. [ANSWERED 2026-06-15]
24. **VFI (#11, was "VFP" in the prompt)** → **Vocal Fry Index**: proportion of voiced frames exhibiting creaky
    pulse clustering [62]. Method = **DeepFry** (Chernyak et al., arXiv 2203.17019),
    paper model (CREAK-220), run in the isolated `deepfry` env via subprocess;
    creak intervals from its `pred-creaky` TextGrid tier → fraction of voiced
    frames in creak. Vendored at third_party/deepfry. [ANSWERED 2026-06-15]
    **VFI=0 is a valid measurement (no fry), not missing.** A voiced utterance
    is never NaN: a DeepFry failure/empty result, or the min-valid-frames guard,
    yields VFI=0 (enforced in `deepfry_creak.compute` + an `extract.py` guard;
    one DeepFry retry absorbs transient failures). VFI is NaN only when there are
    no voiced frames, so it is analysed **raw** — the earlier `log_nonzero`
    transform (0→NaN→log) wrongly dropped the ~489 zero-fry utterances and was
    removed. [REVISED 2026-06-18]
25. **Nasality (#23)** → **A1-P0** (Chen 1997, JASA 102(4):2360-2370), a
    single-channel acoustic nasalization correlate: A1 = peak dB near F1, P0 =
    peak dB in the 200-500 Hz nasal-murmur band, Nasality = A1-P0 (lower = more
    nasal). Chosen over the DL nasalance route (Lozano 2024 needs dual-channel
    Nasometer training + missing lib/ + no pretrained model + Spanish domain).
    Pure numpy/scipy in-env, no model/training. [ANSWERED 2026-06-15]
26. **alpha_ratio (#29)** → **E(1–5 kHz) / E(50–1000 Hz)** [35]. [ANSWERED]
27. **LHR (#30)** → **E(<1 kHz) / E(>3 kHz)** [46] (note the 1–3 kHz gap).
    [ANSWERED]
28. **SPI (#31)** → low/high harmonic energy ratio [54]; bands not in paper →
    MDVP **70–1600 / 1600–4500 Hz** proposed, ❓CONFIRM bands.
29. **GNE (#32)** → Glottal-to-Noise Excitation, periodic vs turbulent energy in
    the **0.3–4 kHz** band [42]. [ANSWERED]
30. **SSPF (#33)** → **Sibilant Spectral Peak Frequency**: centre freq of /s,ʃ/
    fricative noise [53]. ❓CONFLICT: paper ⇒ **sibilant** domain (/s sh z zh/),
    but prompt §4 lists SSPF as "speech, mean". Which domain?
31. **dCPP (#35)** → frame-wise change (delta) in CPP [39]. [ANSWERED]
32. **AMD (#37)** → Amplitude Modulation Depth: strength of slow (**<20 Hz**)
    envelope fluctuation [36]. [ANSWERED] (exact normalization → at impl time.)
33. **speech_rate (#38)** → **syllables per second** [58]. ❓ TIMIT has no syllable
    tier → need a syllabification rule (e.g. count vowel nuclei). Confirm rule.
34. **VOT (#39)** → Voice Onset Time, stop release vs voicing timing [64];
    operationalize from `Xcl`→`X` alignment (sign for prevoiced stops?). [ANSWERED
    def; sign convention ❓]
35. **BGD (#40)** → already standard: mean inter-pause speech-run duration.

### ❗Conflicts the paper surfaced with earlier decisions:
- **C1 (#14 rolloff):** → **keep 85%** (paper's 95% deliberately overridden).
  [ANSWERED 2026-06-15]
- **C2 (SSPF domain):** → **add a sibilant domain** = phones {s, sh, z, zh};
  SSPF computed on sibilant frames. [ANSWERED 2026-06-15]
- **C3 (speech_rate):** → **syllables = vowel-class nuclei** (count vowel-class
  segments); rate = nuclei / duration. [ANSWERED 2026-06-15]
- **C4 (SPI):** ref **Hillenbrand & Houde (1996), JSLHR 39(2):311-321**. Read the
  paper: it does NOT contain "SPI"/"Soft Phonation Index". Its matching spectral-
  energy-ratio measure is **"H/L" = avg energy ≥4 kHz / avg energy <4 kHz** (128-pt
  / 5.1 ms Fourier spectra every 2.56 ms). **Adopted H/L as SPI** (single 4 kHz
  split, high-over-low). [DEFN ANSWERED 2026-06-15]
  Direction confirmed high-over-low (E≥4k / E<4k) per the paper. [ANSWERED]

---
## ✅ CHECKPOINT 1 (CONFIG block) APPROVED — 2026-06-15
All 40 features defined; all parameters pinned (IAIF sub-params + SPI direction
confirmed). Next: implement, run on ONE utterance, then ONE speaker, pausing for
review at each. Do NOT process the full corpus until the user says "create the feats".
Impl-time operational details still to fix from each cited ref: VFI creak detector
[62], Nasality measure [48], AMD normalization [36], VOT sign for prevoiced stops.
