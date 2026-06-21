"""CONFIG block for the TIMIT 40-feature utterance-level extractor.

Every parameter that affects a number lives here, in the open. Nothing in this
file is applied until it has been reviewed and approved (verification protocol,
checkpoint 1).

Markers:
  ❓CONFIRM  — a methodology choice I am NOT certain matches your intent. The
              value present is a *proposal*; see DECISIONS.md for the question.
  🔒FIXED    — pinned by the prompt or by TIMIT itself; not a free choice.

The CONFIG hash written into every output file is computed over the contents of
the CONFIG dataclass instance (see config_hash()), so each result records which
parameter set produced it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────────────────────────────────────────

# 🔒FIXED by user decision (DECISIONS.md #1): the feature tree is written here,
# NOT as a sibling of the TIMIT root. The CLI may override with --out.
DEFAULT_OUTPUT_ROOT = Path("~/data/timit-feats").expanduser()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Signal / framing parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FramingConfig:
    sample_rate_expected: int = 16000          # 🔒FIXED — TIMIT is 16 kHz
    frame_length_ms: float = 25.0              # 🔒CONFIRMED (DECISIONS #2)
    hop_ms: float = 10.0                       # 🔒CONFIRMED (DECISIONS #2)
    window: str = "hann"                       # 🔒CONFIRMED (DECISIONS #3)
    pre_emphasis: float = 0.97                 # 🔒CONFIRMED (DECISIONS #4)
    frame_to_phone: str = "center_sample"      # 🔒FIXED — map frame by its center sample


# ─────────────────────────────────────────────────────────────────────────────
# 2. F0 / source analysis parameters  (Praat / parselmouth)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PitchConfig:
    # Sex-dependent range; sex is read from the speaker-dir initial (M/F).
    sex_dependent_f0: bool = True              # 🔒CONFIRMED sex-dependent (DECISIONS #5)
    f0_floor_male_hz: float = 75.0             # 🔒CONFIRMED (DECISIONS #5b)
    f0_ceiling_male_hz: float = 300.0          # 🔒CONFIRMED (DECISIONS #5b)
    f0_floor_female_hz: float = 100.0          # 🔒CONFIRMED (DECISIONS #5b)
    f0_ceiling_female_hz: float = 500.0        # 🔒CONFIRMED (DECISIONS #5b)

    # F0 extraction: Praat autocorrelation (Boersma 1993), parselmouth
    # `To Pitch (ac)`. Chosen over pYIN: empirically a wash on TIMIT (~8.5 cents
    # median diff, ~equal voicing yield, 0.5% octave disagreement) and Praat-ac
    # keeps ONE pitch estimator consistent with the Praat-based perturbation/
    # voice-quality features. (DECISIONS #5c)
    pitch_method: str = "ac"                   # 🔒CONFIRMED (DECISIONS #5c)
    pitch_time_step_ms: float = 10.0           # 🔒CONFIRMED — match hop (#5c)

    # Praat voicing-decision parameters (ac defaults) — set WHICH frames are voiced
    # and thus the valid-frame counts for every F0-dependent feature (Group 2).
    silence_threshold: float = 0.03            # 🔒CONFIRMED Praat default (#5c)
    voicing_threshold: float = 0.45            # 🔒CONFIRMED Praat default (#5c)
    octave_cost: float = 0.01                  # 🔒CONFIRMED Praat default (#5c)
    octave_jump_cost: float = 0.35             # 🔒CONFIRMED Praat default (#5c)
    voiced_unvoiced_cost: float = 0.14         # 🔒CONFIRMED Praat default (#5c)
    max_candidates: int = 15                   # 🔒CONFIRMED Praat default (#5c)

    # Jitter/shimmer (#2/#3) computed from a CROSS-CORRELATION PointProcess
    # (Praat Voice Report convention). (DECISIONS #5c)
    perturbation_point_process_method: str = "cc"   # 🔒CONFIRMED (DECISIONS #5c)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Formant analysis parameters  (Praat / parselmouth Burg)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FormantConfig:
    # HYBRID estimator (DECISIONS #6/#7 revised by #D). Rationale & precedent:
    # Gowda/Kadiri/Alku, "Refining a Deep Learning-based Formant Tracker using
    # Linear Prediction Methods", Computer Speech & Language 2023 (arXiv 2308.09051).
    #   F1–F4 frequencies          ← DeepFormants (trained on VTR-TIMIT)
    #   F5 frequency + B1–B5        ← order-20 Burg LPC at NATIVE 16 kHz (no downsample)
    #   spurious Burg poles rejected by matching them to the DeepFormants F1–F4.
    method: str = "hybrid_deepformants_burg"   # 🔒CONFIRMED (DECISIONS #6/#7/#D)
    n_formants_reported: int = 5               # 🔒FIXED — report F1..F5 (+B1..B5)

    # ── DeepFormants (source of F1–F4 frequencies) ────────────────────────────
    deepformants_n: int = 4                    # 🔒 DeepFormants yields F1–F4 only
    deepformants_track_hop_ms: float = 10.0    # 🔒 DeepFormants tracking grid (matches our hop)

    # ── Burg LPC (source of F5 + all five bandwidths) ─────────────────────────
    burg_order: int = 20                       # 🔒CONFIRMED order-20 Burg (DECISIONS #D)
    burg_native_rate: bool = True              # 🔒CONFIRMED analyze at native 16 kHz, NO resample
    burg_window_ms: float = 25.0               # 🔒CONFIRMED (DECISIONS #7b)
    # Burg uses FramingConfig.pre_emphasis (0.97); this is our own LPC, not Praat's.

    # ── Pole → formant assignment / spurious-peak rejection ───────────────────
    sex_dependent_ceiling: bool = True         # 🔒CONFIRMED (DECISIONS #6)
    formant_ceiling_male_hz: float = 5000.0    # 🔒CONFIRMED max accepted formant freq, male
    formant_ceiling_female_hz: float = 5500.0  # 🔒CONFIRMED max accepted formant freq, female
    df_match_tolerance_hz: float = 300.0       # 🔒CONFIRMED (DECISIONS #Da) max |Burg_pole −
                                               #   DeepFormants_Fk| to accept a pole as real F1..F4
    f5_max_bandwidth_hz: float = 1000.0        # 🔒CONFIRMED (DECISIONS #Db) candidate F5 pole must
                                               #   be narrower than this to count as a resonance
    # F5 selection rule (🔒 given the above): among Burg poles above matched-F4 and
    # below the sex ceiling, with bandwidth < f5_max_bandwidth_hz, take the LOWEST-
    # frequency such pole as F5; its bandwidth is B5. B1–B4 are the bandwidths of the
    # Burg poles matched to DeepFormants F1–F4. If no qualifying F5 pole → F5,B5 = NaN.


# ─────────────────────────────────────────────────────────────────────────────
# 4. Aggregation / validity guards
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AggregationConfig:
    # Applies ONLY to frame-level features (#1-37): the count is of valid in-domain
    # frames summed over the WHOLE utterance (not per segment/event). Below it → NaN.
    # Alignment-native features (#38 speech_rate, #39 VOT, #40 BGD) are NOT subject
    # to this guard; they use event-based validity (e.g. VOT → NaN iff no identifiable
    # stop release). See DECISIONS #8 and #38-40.
    min_valid_frames: int = 5                  # 🔒CONFIRMED (DECISIONS #8)
    random_seed: int = 0                       # 🔒FIXED — determinism (no RNG should be needed)


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Spectral feature parameters (the confirmed "standard" ones)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpectralConfig:
    rolloff_percent: float = 0.85              # 🔒CONFIRMED 0.85 (DECISIONS C1; paper's 95%
                                              #   deliberately overridden by the user)
    moments_domain: str = "speech"            # 🔒CONFIRMED (DECISIONS #15) all-speech frames
    cpp_method: str = "praat_cpps"            # 🔒CONFIRMED (DECISIONS #16) Praat PowerCepstrogram

    # Band definitions from the paper (Table 1). (Hz)
    alpha_low_band: tuple = (50.0, 1000.0)     # alpha_ratio denominator [35]
    alpha_high_band: tuple = (1000.0, 5000.0)  # alpha_ratio numerator; ratio = high/low
    lhr_low_band: tuple = (0.0, 1000.0)        # LHR numerator: E(<1 kHz) [46]
    lhr_high_band: tuple = (3000.0, 8000.0)    # LHR denominator: E(>3 kHz); ratio = low/high
    gne_band: tuple = (300.0, 4000.0)          # GNE 0.3-4 kHz [42]
    amd_mod_max_hz: float = 20.0               # AMD: slow envelope modulation < 20 Hz [36]
    # SPI (Soft Phonation Index) — ref: Hillenbrand & Houde (1996), JSLHR 39(2):311-321.
    # NB: that paper does NOT use the term "SPI". Its matching spectral-energy-ratio
    # measure is "H/L" = avg energy >=4 kHz over avg energy <4 kHz (from 128-pt /
    # 5.1 ms Fourier spectra every 2.56 ms). Adopting H/L as SPI per the cited ref.
    spi_crossover_hz: float = 4000.0           # 🔒CONFIRMED H&H single 4 kHz split (C4)
    spi_direction: str = "high_over_low"       # 🔒CONFIRMED E(>=4k)/E(<4k) per H&H (C4)


# ─────────────────────────────────────────────────────────────────────────────
# 4c. Glottal-source front-end (shared by GCT, CQ, NAQ, MFDR, SQ)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GlottalSourceConfig:
    # No EGG in TIMIT → estimate the glottal flow by inverse filtering, then read
    # the cycle parameters off the flow and its derivative. Method = IAIF
    # (Iterative Adaptive Inverse Filtering, Alku 1992). Shared by:
    #   GCT (#4) closed-phase duration (ms), CQ (#5) closed/period ratio,
    #   MFDR (#6) max flow declination rate, SQ (#7) speed quotient,
    #   NAQ (#8) normalized amplitude quotient.
    method: str = "iaif"                   # 🔒CONFIRMED (DECISIONS #17/#18)
    # IAIF sub-parameters (affect the flow estimate → all five features):
    vt_lpc_order: int = 18                 # 🔒CONFIRMED vocal-tract LPC order (~Fs/1000+2 @16k)
    glottal_lpc_order: int = 4             # 🔒CONFIRMED glottal (source) LPC order g
    n_iterations: int = 2                  # 🔒CONFIRMED IAIF iterations
    lip_radiation_coef: float = 0.99       # 🔒CONFIRMED leaky-integration / lip-radiation coef
    highpass_hz: float = 40.0              # 🔒CONFIRMED pre-analysis high-pass cutoff
    # Closed/open phase boundaries (GCI/GOI) for GCT, CQ, SQ:
    gci_source: str = "flow_derivative"    # 🔒CONFIRMED GCI/GOI from IAIF flow derivative


# ─────────────────────────────────────────────────────────────────────────────
# 5. TIMIT 61-phone ARPABET → phone-class map
#    Classes drive the per-feature segment domains in §7.
#    🔒CONFIRMED mapping (DECISIONS #9-13).
# ─────────────────────────────────────────────────────────────────────────────

# Class vocabulary:
#   silence, vowel, nasal, liquid, glide,
#   voiced_obstruent, voiceless_obstruent, closure, other
PHONE_CLASS: dict[str, str] = {
    # ── silence ──────────────────────────────────────────────────────────────
    "h#": "silence", "pau": "silence", "epi": "silence",
    # ── vowels (monophthongs, diphthongs, reduced) ─────────────────────────────
    "iy": "vowel", "ih": "vowel", "eh": "vowel", "ey": "vowel", "ae": "vowel",
    "aa": "vowel", "aw": "vowel", "ay": "vowel", "ah": "vowel", "ao": "vowel",
    "oy": "vowel", "ow": "vowel", "uh": "vowel", "uw": "vowel", "ux": "vowel",
    "er": "vowel", "ax": "vowel", "ix": "vowel", "axr": "vowel", "ax-h": "vowel",
    # ── nasals ────────────────────────────────────────────────────────────────
    "m": "nasal", "n": "nasal", "ng": "nasal",
    "em": "nasal", "en": "nasal", "eng": "nasal", "nx": "nasal",
    # ── liquids ───────────────────────────────────────────────────────────────
    "l": "liquid", "el": "liquid", "r": "liquid",
    # ── glides ────────────────────────────────────────────────────────────────
    "w": "glide", "y": "glide",
    "hh": "glide",  # 🔒CONFIRMED glide (DECISIONS #10)
    "hv": "glide",  # 🔒CONFIRMED glide (DECISIONS #10)
    # ── voiced obstruents ─────────────────────────────────────────────────────
    "b": "voiced_obstruent", "d": "voiced_obstruent", "g": "voiced_obstruent",
    "jh": "voiced_obstruent",
    "z": "voiced_obstruent", "zh": "voiced_obstruent",
    "v": "voiced_obstruent", "dh": "voiced_obstruent",
    "dx": "voiced_obstruent",  # 🔒CONFIRMED flap (DECISIONS #11)
    # ── voiceless obstruents ──────────────────────────────────────────────────
    "p": "voiceless_obstruent", "t": "voiceless_obstruent", "k": "voiceless_obstruent",
    "ch": "voiceless_obstruent",
    "s": "voiceless_obstruent", "sh": "voiceless_obstruent",
    "f": "voiceless_obstruent", "th": "voiceless_obstruent",
    # ── stop closures (DECISIONS #9, option 1) ────────────────────────────────
    # Voiceless closures are silent → 'closure' (excluded from acoustic features;
    # used only to locate stop releases for VOT). Voiced closures carry the
    # low-frequency "voice bar", so they are voiced_obstruent → they enter the
    # voiced + speech domains (and jitter/shimmer) but NOT sonorant (no oral
    # formants during the occlusion).
    "bcl": "voiced_obstruent", "dcl": "voiced_obstruent", "gcl": "voiced_obstruent",
    "pcl": "closure", "tcl": "closure", "kcl": "closure",
    # ── other ─────────────────────────────────────────────────────────────────
    "q": "voiceless_obstruent",   # 🔒CONFIRMED glottal stop (DECISIONS #12)
}

# Which classes count as members of each domain used by §7.
# 🔒CONFIRMED set memberships (DECISIONS #13).
VOICED_CLASSES = ("vowel", "nasal", "liquid", "glide", "voiced_obstruent")
SONORANT_CLASSES = ("vowel", "nasal", "liquid", "glide")
SPEECH_CLASSES = (  # all non-silence; closures (pcl/tcl/kcl) excluded via SILENCE_CLASSES
    "vowel", "nasal", "liquid", "glide",
    "voiced_obstruent", "voiceless_obstruent",
)
SILENCE_CLASSES = ("silence", "closure")  # 🔒CONFIRMED closure∈silence (DECISIONS #13)

# Sibilant fricatives — a PHONE-level set (not a class), the domain for SSPF (#33).
# 🔒CONFIRMED add sibilant domain (DECISIONS C2).
SIBILANT_PHONES = ("s", "sh", "z", "zh")


# ─────────────────────────────────────────────────────────────────────────────
# 6. The 40 features — fixed order, level, segment domain, aggregation.
#    Order and aggregation are 🔒FIXED by the prompt §4. Domains follow §3.
#    `status` flags features whose DEFINITION still needs confirmation before
#    any implementation (prompt §4 final paragraph).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureSpec:
    index: int
    name: str
    level: str          # "frame" | "utterance"
    domain: str         # "voiced" | "sonorant" | "speech" | "sibilant" | "alignment"
    aggregation: str    # "median" | "mean" | "sd_semitone" | "flux_mean" | native
    status: str         # "standard" | "needs_definition"
    note: str = ""


FEATURE_ORDER: tuple[FeatureSpec, ...] = (
    FeatureSpec(1,  "F0",                "frame", "voiced",   "median",      "standard",         "Praat pitch"),
    FeatureSpec(2,  "jitter",            "frame", "voiced",   "mean",        "standard",         "Praat local jitter"),
    FeatureSpec(3,  "shimmer",           "frame", "voiced",   "mean",        "standard",         "Praat local shimmer"),
    FeatureSpec(4,  "GCT",               "frame", "voiced",   "mean",        "standard",         "Glottal Closure Time = IAIF closed-phase duration (ms) (#17)"),
    FeatureSpec(5,  "CQ",                "frame", "voiced",   "mean",        "standard",         "Closed Quotient = IAIF closed/period ratio (#18)"),
    FeatureSpec(6,  "MFDR",              "frame", "voiced",   "mean",        "standard",         "max |min(d/dt IAIF flow)| at GCI, per cycle (#19)"),
    FeatureSpec(7,  "SQ",                "frame", "voiced",   "mean",        "standard",         "Speed Quotient = opening/closing phase of IAIF flow (#20)"),
    FeatureSpec(8,  "NAQ",               "frame", "voiced",   "mean",        "standard",         "Alku 2002: f_AC/(d_peak*T0) from IAIF flow (#21)"),
    FeatureSpec(9,  "SHR",               "frame", "voiced",   "mean",        "standard",         "Sun 2002 SHRP subharmonic/harmonic ratio (#22)"),
    FeatureSpec(10, "IHI",               "frame", "voiced",   "mean",        "standard",         "Inharmonicity Index: mean deviation of harmonics from k*F0 [44]"),
    FeatureSpec(11, "VFI",               "frame", "voiced",   "mean",        "standard",         "Vocal Fry Index: proportion of voiced frames creaky, via DeepFry (arXiv 2203.17019) [62]"),
    FeatureSpec(12, "semitone_SD_F0",    "frame", "voiced",   "sd_semitone", "standard",         "SD of F0 in semitones"),
    FeatureSpec(13, "F1",                "frame", "sonorant", "median",      "standard",         "DeepFormants (VTR-TIMIT)"),
    FeatureSpec(14, "F2",                "frame", "sonorant", "median",      "standard",         "DeepFormants (VTR-TIMIT)"),
    FeatureSpec(15, "F3",                "frame", "sonorant", "median",      "standard",         "DeepFormants (VTR-TIMIT)"),
    FeatureSpec(16, "F4",                "frame", "sonorant", "median",      "standard",         "DeepFormants (VTR-TIMIT)"),
    FeatureSpec(17, "F5",                "frame", "sonorant", "median",      "standard",         "Burg order-20 pole above DL-matched F4"),
    FeatureSpec(18, "B1",                "frame", "sonorant", "median",      "standard",         "Burg BW of pole matched to DF F1"),
    FeatureSpec(19, "B2",                "frame", "sonorant", "median",      "standard",         "Burg BW of pole matched to DF F2"),
    FeatureSpec(20, "B3",                "frame", "sonorant", "median",      "standard",         "Burg BW of pole matched to DF F3"),
    FeatureSpec(21, "B4",                "frame", "sonorant", "median",      "standard",         "Burg BW of pole matched to DF F4"),
    FeatureSpec(22, "B5",                "frame", "sonorant", "median",      "standard",         "Burg BW of the F5 pole"),
    FeatureSpec(23, "Nasality",          "frame", "sonorant", "mean",        "standard",         "A1-P0 acoustic nasalization (Chen 1997); single-channel [48]"),
    FeatureSpec(24, "spectral_skewness", "frame", "speech",   "mean",        "standard",         "scipy moment of power spectrum"),
    FeatureSpec(25, "spectral_kurtosis", "frame", "speech",   "mean",        "standard",         "scipy moment of power spectrum"),
    FeatureSpec(26, "spectral_entropy",  "frame", "speech",   "mean",        "standard",         "Shannon entropy of normalized spectrum"),
    FeatureSpec(27, "spectral_rolloff",  "frame", "speech",   "mean",        "standard",         "librosa rolloff @ 85% (DECISIONS C1; paper's 95% overridden)"),
    FeatureSpec(28, "spectral_flux",     "frame", "speech",   "flux_mean",   "standard",         "mean frame-to-frame flux"),
    FeatureSpec(29, "alpha_ratio",       "frame", "speech",   "mean",        "standard",         "Alpha Ratio = E(1-5kHz)/E(50-1000Hz) [35]"),
    FeatureSpec(30, "LHR",               "frame", "speech",   "mean",        "standard",         "Low/High = E(<1kHz)/E(>3kHz) [46]"),
    FeatureSpec(31, "SPI",               "frame", "speech",   "mean",        "standard",         "SPI = H&H 1996 'H/L' = E(>=4kHz)/E(<4kHz) (C4)"),
    FeatureSpec(32, "GNE",               "frame", "voiced",   "mean",        "standard",         "Glottal-to-Noise Excitation, 0.3-4 kHz band [42]"),
    FeatureSpec(33, "SSPF",              "frame", "sibilant", "mean",        "standard",         "Sibilant Spectral Peak Freq of /s sh z zh/ [53] (DECISIONS C2)"),
    FeatureSpec(34, "CPP",               "frame", "voiced",   "mean",        "standard",         "Praat PowerCepstrogram CPPS, smoothed (#16)"),
    FeatureSpec(35, "dCPP",              "frame", "voiced",   "mean",        "standard",         "frame-wise change (delta) in CPP [39]"),
    FeatureSpec(36, "RMS",               "frame", "speech",   "mean",        "standard",         "frame RMS energy"),
    FeatureSpec(37, "AMD",               "frame", "speech",   "mean",        "standard",         "Amplitude Modulation Depth: slow (<20 Hz) envelope fluctuation [36]"),
    FeatureSpec(38, "speech_rate",       "utterance", "alignment", "native", "standard",         "syllables/sec [58]; syllables = vowel-class nuclei (DECISIONS C3)"),
    FeatureSpec(39, "VOT",               "utterance", "alignment", "native", "standard",         "Voice Onset Time: stop release→voicing [64]"),
    FeatureSpec(40, "BGD",               "utterance", "alignment", "native", "standard",         "mean inter-pause speech-run duration"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURE_ORDER)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level CONFIG and hashing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NasalityConfig:
    # Nasality (#23) = A1 - P0 (Chen 1997), a single-channel acoustic nasalization
    # correlate. A1 = spectral peak amplitude (dB) near F1; P0 = peak amplitude (dB)
    # in the low-frequency nasal-murmur band. Lower A1-P0 ⇒ more nasal.
    nasal_band: tuple = (200.0, 500.0)         # 🔒CONFIRMED P0 nasal-peak search band (Hz)
    a1_rel_halfwidth: float = 0.2              # 🔒 A1 search window = F1*(1 ± this)
    sign: str = "a1_minus_p0"                  # 🔒CONFIRMED report A1-P0 (lower = more nasal)


@dataclass(frozen=True)
class CreakConfig:
    # VFI (#11) via the vendored DeepFry creak detector (arXiv 2203.17019).
    model: str = "paper"        # 🔒CONFIRMED 'paper' (CREAK-220) — vs 'both_datasets' (CREAK-74)
    creak_mark: str = "c"       # 🔒 mark used in DeepFry's pred-creaky TextGrid tier
    creak_tier: str = "pred-creaky"


@dataclass(frozen=True)
class TimingConfig:
    # Alignment-native features (speech_rate, VOT, BGD).
    syllable_unit: str = "vowel_nuclei"        # 🔒CONFIRMED (DECISIONS C3) count vowel-class
                                               #   segments as syllables for speech_rate


@dataclass(frozen=True)
class Config:
    framing: FramingConfig = field(default_factory=FramingConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    glottal: GlottalSourceConfig = field(default_factory=GlottalSourceConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    formant: FormantConfig = field(default_factory=FormantConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    creak: CreakConfig = field(default_factory=CreakConfig)
    nasality: NasalityConfig = field(default_factory=NasalityConfig)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["phone_class"] = PHONE_CLASS
        d["voiced_classes"] = list(VOICED_CLASSES)
        d["sonorant_classes"] = list(SONORANT_CLASSES)
        d["speech_classes"] = list(SPEECH_CLASSES)
        d["silence_classes"] = list(SILENCE_CLASSES)
        d["sibilant_phones"] = list(SIBILANT_PHONES)
        d["feature_names"] = list(FEATURE_NAMES)
        return d

    def config_hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


CONFIG = Config()
