"""Unit tests for the config block and the IO + framing/masking foundation."""
import os
from pathlib import Path

import numpy as np
import pytest

from timit_features.config import (
    CONFIG, FEATURE_NAMES, FEATURE_ORDER, PHONE_CLASS,
    VOICED_CLASSES, SONORANT_CLASSES, SPEECH_CLASSES, SIBILANT_PHONES,
)
from timit_features.io_timit import load_utterance
from timit_features.framing import build_frames
from timit_features import (features_spectral, features_praat, features_alignment,
                            features_glottal, features_harmonic, features_formant)
from timit_features.aggregate import aggregate_frame_feature

TIMIT_ROOT = Path(os.environ.get("TIMIT_ROOT", "/home/bhiksha/data/timit/TIMIT"))
SA1 = TIMIT_ROOT / "TRAIN" / "DR1" / "FCJF0" / "SA1.WAV"
need_corpus = pytest.mark.skipif(not SA1.exists(), reason="TIMIT corpus not present")


# ── CONFIG ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_forty_features_unique(self):
        assert len(FEATURE_NAMES) == 40
        assert len(set(FEATURE_NAMES)) == 40

    def test_all_features_defined(self):
        undefined = [f.name for f in FEATURE_ORDER if f.status == "needs_definition"]
        assert undefined == []

    def test_61_phones_mapped(self):
        assert len(PHONE_CLASS) == 61

    def test_domain_nesting(self):
        # sonorant ⊆ voiced ⊆ speech (as class sets)
        assert set(SONORANT_CLASSES) <= set(VOICED_CLASSES) <= set(SPEECH_CLASSES)

    def test_sibilants_are_obstruents(self):
        for p in SIBILANT_PHONES:
            assert PHONE_CLASS[p].endswith("obstruent")

    def test_config_hash_deterministic(self):
        h = CONFIG.config_hash()
        assert h == CONFIG.config_hash()
        assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)


# ── IO ────────────────────────────────────────────────────────────────────────

@need_corpus
class TestIO:
    def test_load_sa1_metadata(self):
        u = load_utterance(SA1, TIMIT_ROOT)
        assert u.decode_ok and u.sample_rate == 16000
        assert u.speaker_id == "FCJF0" and u.sex == "F"
        assert u.split == "TRAIN" and u.dialect_region == "DR1"
        assert u.basename == "SA1"
        assert "She had" in u.text
        assert len(u.phones) == 37 and len(u.words) == 11
        assert u.data is not None and u.n_samples == u.data.shape[0]

    def test_missing_file_does_not_raise(self):
        u = load_utterance(TIMIT_ROOT / "TRAIN/DR1/FCJF0/NOPE.WAV", TIMIT_ROOT)
        assert u.decode_ok is False and u.data is None and u.error


# ── Framing / masking ──────────────────────────────────────────────────────────

@need_corpus
class TestFraming:
    def _frames(self):
        u = load_utterance(SA1, TIMIT_ROOT)
        return build_frames(u.n_samples, u.phones, CONFIG)

    def test_array_lengths_consistent(self):
        fr = self._frames()
        n = fr.n_frames
        assert n > 0
        for arr in (fr.center_samples, fr.start_samples, fr.phone_class,
                    fr.voiced, fr.sonorant, fr.speech, fr.sibilant):
            assert len(arr) == n
        assert len(fr.phone) == n

    def test_per_frame_domain_nesting(self):
        fr = self._frames()
        # every sonorant frame is voiced; every voiced frame is speech
        assert np.all(fr.speech[fr.voiced])
        assert np.all(fr.voiced[fr.sonorant])
        assert np.all(fr.speech[fr.sibilant])

    def test_silence_and_closure_excluded(self):
        fr = self._frames()
        excl = np.array([c in ("silence", "closure") for c in fr.phone_class])
        assert not np.any(fr.voiced[excl])
        assert not np.any(fr.speech[excl])

    def test_masks_match_phone_class(self):
        fr = self._frames()
        for i, c in enumerate(fr.phone_class):
            assert fr.voiced[i] == (c in VOICED_CLASSES)
            assert fr.speech[i] == (c in SPEECH_CLASSES)

    def test_determinism(self):
        u = load_utterance(SA1, TIMIT_ROOT)
        a = build_frames(u.n_samples, u.phones, CONFIG)
        b = build_frames(u.n_samples, u.phones, CONFIG)
        assert np.array_equal(a.center_samples, b.center_samples)
        assert np.array_equal(a.voiced, b.voiced)
        assert a.phone == b.phone


# ── Aggregation ─────────────────────────────────────────────────────────────────

class TestAggregate:
    def test_guard_below_min_returns_nan(self):
        vals = np.array([100.0, 110.0, 120.0])          # only 3 valid
        mask = np.ones(3, dtype=bool)
        v, n = aggregate_frame_feature(vals, mask, "mean", CONFIG)
        assert n == 3 and np.isnan(v)

    def test_mean_and_median(self):
        vals = np.arange(10.0)
        mask = np.ones(10, dtype=bool)
        v, n = aggregate_frame_feature(vals, mask, "mean", CONFIG)
        assert n == 10 and v == pytest.approx(4.5)
        v, _ = aggregate_frame_feature(vals, mask, "median", CONFIG)
        assert v == pytest.approx(4.5)

    def test_nans_excluded_from_count(self):
        vals = np.array([1, 2, np.nan, 4, np.nan, 6, 7.0])
        mask = np.ones(7, dtype=bool)
        v, n = aggregate_frame_feature(vals, mask, "mean", CONFIG)
        assert n == 5 and v == pytest.approx(4.0)


# ── Spectral feature group ──────────────────────────────────────────────────────

@need_corpus
class TestSpectral:
    def _setup(self):
        u = load_utterance(SA1, TIMIT_ROOT)
        fr = build_frames(u.n_samples, u.phones, CONFIG)
        vals = features_spectral.compute(u.data, CONFIG)
        return u, fr, vals

    def test_arrays_aligned_with_frames(self):
        _, fr, vals = self._setup()
        for name, arr in vals.items():
            assert len(arr) == fr.n_frames, name

    def test_speech_frame_values_finite(self):
        _, fr, vals = self._setup()
        for name in ("spectral_skewness", "spectral_rolloff", "alpha_ratio", "LHR", "SPI", "RMS"):
            assert np.all(np.isfinite(vals[name][fr.speech])), name

    def test_rolloff_within_nyquist(self):
        _, fr, vals = self._setup()
        ro = vals["spectral_rolloff"][fr.speech]
        assert np.all((ro >= 0) & (ro <= 8000))

    def test_determinism(self):
        u, _, v1 = self._setup()
        v2 = features_spectral.compute(u.data, CONFIG)
        for name in v1:
            assert np.array_equal(v1[name], v2[name], equal_nan=True)


# ── Praat feature group ─────────────────────────────────────────────────────────

@need_corpus
class TestPraat:
    def _vals(self):
        u = load_utterance(SA1, TIMIT_ROOT)
        fr = build_frames(u.n_samples, u.phones, CONFIG)
        return u, fr, features_praat.compute(u.data, fr, u.phones, u.sex, CONFIG)

    def test_keys_and_lengths(self):
        u, fr, v = self._vals()
        assert set(v) == {"F0", "semitone_SD_F0", "jitter", "shimmer", "CPP"}
        for arr in v.values():
            assert len(arr) == fr.n_frames

    def test_f0_plausible_for_female(self):
        u, fr, v = self._vals()
        f0 = v["F0"][fr.voiced]
        f0 = f0[np.isfinite(f0)]
        assert f0.size >= CONFIG.aggregation.min_valid_frames
        assert 100.0 <= np.median(f0) <= 500.0     # FCJF0 is female

    def test_f0_within_configured_bounds(self):
        u, fr, v = self._vals()
        f0 = v["F0"][np.isfinite(v["F0"])]
        assert np.all((f0 >= CONFIG.pitch.f0_floor_female_hz - 1) &
                      (f0 <= CONFIG.pitch.f0_ceiling_female_hz + 1))


# ── Alignment-native group ──────────────────────────────────────────────────────

@need_corpus
class TestAlignment:
    def _vals(self):
        u = load_utterance(SA1, TIMIT_ROOT)
        return features_alignment.compute(u.phones, CONFIG)

    def test_keys(self):
        assert set(self._vals()) == {"speech_rate", "VOT", "BGD"}

    def test_speech_rate_reasonable(self):
        v, n = self._vals()["speech_rate"]
        assert n > 0 and 1.0 <= v <= 12.0     # syllables/sec

    def test_vot_and_bgd_positive_or_nan(self):
        vals = self._vals()
        for name in ("VOT", "BGD"):
            v, n = vals[name]
            assert (np.isnan(v) and n == 0) or (v > 0 and n > 0)

    def test_no_stop_releases_gives_nan(self):
        from timit_features.io_timit import Segment
        # a silence-only alignment → VOT NaN, speech_rate NaN
        phones = [Segment(0, 16000, "h#")]
        vals = features_alignment.compute(phones, CONFIG)
        assert np.isnan(vals["VOT"][0]) and vals["VOT"][1] == 0
        assert np.isnan(vals["speech_rate"][0])


# ── IAIF glottal group ──────────────────────────────────────────────────────────

@need_corpus
class TestGlottal:
    def _vals(self):
        u = load_utterance(SA1, TIMIT_ROOT)
        fr = build_frames(u.n_samples, u.phones, CONFIG)
        return u, fr, features_glottal.compute(u.data, fr, u.phones, u.sex, CONFIG)

    def test_keys_lengths(self):
        u, fr, v = self._vals()
        assert set(v) == {"GCT", "CQ", "NAQ", "MFDR", "SQ"}
        for arr in v.values():
            assert len(arr) == fr.n_frames

    def test_some_voiced_values_present(self):
        u, fr, v = self._vals()
        for name in ("GCT", "CQ", "NAQ", "MFDR"):
            finite = np.isfinite(v[name][fr.voiced]).sum()
            assert finite >= CONFIG.aggregation.min_valid_frames, name

    def test_cq_in_unit_interval(self):
        u, fr, v = self._vals()
        cq = v["CQ"][np.isfinite(v["CQ"])]
        assert np.all((cq >= 0) & (cq <= 1))

    def test_determinism(self):
        u, fr, v1 = self._vals()
        v2 = features_glottal.compute(u.data, fr, u.phones, u.sex, CONFIG)
        for k in v1:
            assert np.array_equal(v1[k], v2[k], equal_nan=True)


# ── Harmonic / voice-quality group ──────────────────────────────────────────────

@need_corpus
class TestHarmonic:
    def _vals(self):
        u = load_utterance(SA1, TIMIT_ROOT)
        fr = build_frames(u.n_samples, u.phones, CONFIG)
        pv = features_praat.compute(u.data, fr, u.phones, u.sex, CONFIG)
        v = features_harmonic.compute(u.data, fr, pv["F0"], pv["CPP"], u.phones, CONFIG)
        return u, fr, v

    def test_keys(self):
        _, _, v = self._vals()
        # VFI and Nasality are produced by their own modules (deepfry_creak,
        # features_nasality) and filled in by extract.py — not by this group.
        assert set(v) == {"IHI", "SHR", "GNE", "dCPP", "AMD"}

    def test_implemented_features_present(self):
        _, fr, v = self._vals()
        for name in ("IHI", "SHR", "GNE", "dCPP", "AMD"):
            assert np.isfinite(v[name]).sum() >= CONFIG.aggregation.min_valid_frames, name

    def test_gne_in_unit_range(self):
        _, _, v = self._vals()
        g = v["GNE"][np.isfinite(v["GNE"])]
        assert np.all((g >= -1.01) & (g <= 1.01))


# ── Formant hybrid (Burg/poles fast; DeepFormants integration slow) ─────────────

class TestBurg:
    def test_burg_coeffs(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(400)
        a = features_formant.burg(x, 20)
        assert len(a) == 21 and a[0] == 1.0

    def test_poles_recover_synthetic_formants(self):
        sr = 16000
        t = np.arange(0.05 * sr) / sr
        # two damped resonances at 700 and 1800 Hz
        x = (np.exp(-80 * t) * np.sin(2 * np.pi * 700 * t) +
             np.exp(-120 * t) * np.sin(2 * np.pi * 1800 * t))
        x = np.tile(x, 4)
        f, b = features_formant.poles(features_formant.burg(x, 20), sr)
        assert np.any(np.abs(f - 700) < 80) and np.any(np.abs(f - 1800) < 100)


_DF_ENV = Path.home() / "miniconda3" / "envs" / "deepformants"

@need_corpus
@pytest.mark.skipif(not _DF_ENV.exists(), reason="deepformants env not present")
class TestFormant:
    def test_formant_ranges(self):
        u = load_utterance(SA1, TIMIT_ROOT)
        fr = build_frames(u.n_samples, u.phones, CONFIG)
        v = features_formant.compute(u.data, fr, u.phones, u.sex, CONFIG)
        assert set(v) == {f"F{i}" for i in range(1, 6)} | {f"B{i}" for i in range(1, 6)}
        f1 = v["F1"][np.isfinite(v["F1"])]
        assert f1.size >= CONFIG.aggregation.min_valid_frames
        assert 200 <= np.median(f1) <= 1200          # F1 plausible
        f5 = v["F5"][np.isfinite(v["F5"])]
        assert np.all(f5 <= CONFIG.formant.formant_ceiling_female_hz + 1)


# ── Full per-utterance extraction (checkpoint 2) ────────────────────────────────

class TestExtract:
    def test_missing_file_all_nan(self):
        from timit_features.extract import extract_utterance
        r = extract_utterance(TIMIT_ROOT / "TRAIN/DR1/FCJF0/NOPE.WAV", TIMIT_ROOT)
        assert r.decode_ok is False
        assert len(r.vector()) == 40 and np.all(np.isnan(r.vector()))
        assert all(c == 0 for c in r.coverage.values())


_DF_ENV2 = Path.home() / "miniconda3" / "envs" / "deepformants"

@need_corpus
@pytest.mark.skipif(not _DF_ENV2.exists(), reason="deepformants env not present")
class TestExtractFull:
    def test_sa1_vector_and_determinism(self):
        from timit_features.extract import extract_utterance
        r1 = extract_utterance(SA1, TIMIT_ROOT)
        assert r1.decode_ok and len(r1.vector()) == 40
        nonnan = np.isfinite(r1.vector()).sum()
        assert nonnan == 40                       # all 40 features compute
        assert np.isfinite(r1.features["VFI"]) and np.isfinite(r1.features["Nasality"])
        r2 = extract_utterance(SA1, TIMIT_ROOT)
        assert np.array_equal(r1.vector(), r2.vector(), equal_nan=True)


# ── VFI via DeepFry (integration; skips without the deepfry env) ─────────────────

_DFRY_ENV = Path.home() / "miniconda3" / "envs" / "deepfry"

@need_corpus
@pytest.mark.skipif(not _DFRY_ENV.exists(), reason="deepfry env not present")
class TestDeepFryVFI:
    def test_vfi_proportion_in_unit_interval(self):
        from timit_features import deepfry_creak
        u = load_utterance(SA1, TIMIT_ROOT)
        fr = build_frames(u.n_samples, u.phones, CONFIG)
        v = deepfry_creak.compute(u.data, fr, CONFIG)
        assert len(v) == fr.n_frames
        voiced_vals = v[fr.voiced]
        finite = voiced_vals[np.isfinite(voiced_vals)]
        assert finite.size >= CONFIG.aggregation.min_valid_frames
        assert np.all((finite == 0.0) | (finite == 1.0))          # per-frame indicator
        assert 0.0 <= float(np.mean(finite)) <= 1.0               # VFI = proportion


# ── Nasality via A1-P0 (Chen 1997) ──────────────────────────────────────────────

@need_corpus
class TestNasality:
    def test_a1_p0_finite_on_sonorant(self):
        from timit_features import features_nasality, features_formant
        u = load_utterance(SA1, TIMIT_ROOT)
        fr = build_frames(u.n_samples, u.phones, CONFIG)
        # use a cheap synthetic F1 (constant) to avoid the DeepFormants subprocess here
        f1 = np.where(fr.sonorant, 600.0, np.nan)
        v = features_nasality.compute(u.data, fr, f1, CONFIG)
        assert len(v) == fr.n_frames
        finite = v[np.isfinite(v)]
        assert finite.size >= CONFIG.aggregation.min_valid_frames
        assert np.all(np.isfinite(v[~fr.sonorant]) == False)   # only sonorant frames valued

    def test_determinism(self):
        from timit_features import features_nasality
        u = load_utterance(SA1, TIMIT_ROOT)
        fr = build_frames(u.n_samples, u.phones, CONFIG)
        f1 = np.where(fr.sonorant, 600.0, np.nan)
        a = features_nasality.compute(u.data, fr, f1, CONFIG)
        b = features_nasality.compute(u.data, fr, f1, CONFIG)
        assert np.array_equal(a, b, equal_nan=True)


# ── Writers / manifest (fast; no subprocess) ────────────────────────────────────

class TestWriters:
    def _rec(self, name, val):
        from timit_features.extract import UtteranceRecord
        return UtteranceRecord(
            basename=name, rel_path=f"TRAIN/DR1/SPK/{name}.WAV", speaker_id="SPK",
            sex="F", dialect_region="DR1", split="TRAIN", sample_rate=16000,
            duration=2.0, config_hash=CONFIG.config_hash(), decode_ok=True,
            features={n: val for n in FEATURE_NAMES},
            coverage={n: 10 for n in FEATURE_NAMES})

    def test_dataframe_columns_and_rows(self):
        from timit_features import writers
        df = writers.build_dataframe([self._rec("B", 1.0), self._rec("A", 2.0)])
        assert list(df.columns) == writers.COLUMNS
        assert len(df) == 2
        assert list(df["basename"]) == ["A", "B"]      # sorted by rel_path

    def test_manifest_counts(self, tmp_path):
        from timit_features import writers
        recs = [self._rec("A", 1.0), self._rec("B", float("nan"))]
        recs[1].decode_ok = False
        writers.write_manifest(recs, CONFIG, tmp_path)
        import json
        m = json.loads((tmp_path / "MANIFEST.json").read_text())
        assert m["utterances_seen"] == 2 and m["utterances_decoded"] == 1
        assert m["utterances_failed_decode"] == 1


# ── Protocol self-checks (synthetic; need deepformants+deepfry envs) ─────────────

_ENVS_OK = (Path.home() / "miniconda3/envs/deepformants").exists() and \
           (Path.home() / "miniconda3/envs/deepfry").exists()

@pytest.mark.skipif(not _ENVS_OK, reason="deepformants/deepfry envs not present")
class TestSelfChecks:
    def _make(self, tmp, phn, data):
        import soundfile as sf
        d = tmp / "TRAIN" / "DR1" / "FXX0"; d.mkdir(parents=True)
        base = d / "SX1"
        sf.write(str(base) + ".WAV", data.astype(np.float32), 16000, subtype="PCM_16")
        (d / "SX1.PHN").write_text(phn); (d / "SX1.WRD").write_text(""); (d / "SX1.TXT").write_text("0 1 x")
        return str(base) + ".WAV"

    def test_all_silence_all_nan(self, tmp_path):
        from timit_features.extract import extract_utterance
        n = 16000 * 2
        wav = self._make(tmp_path, f"0 {n} h#\n", np.zeros(n))
        r = extract_utterance(wav, tmp_path)
        frame_feats = [f.name for f in FEATURE_ORDER if f.level == "frame"]
        assert all(not np.isfinite(r.features[n_]) for n_ in frame_feats)
        assert all(r.coverage[n_] == 0 for n_ in frame_feats)

    def test_no_voiced_frames(self, tmp_path):
        from timit_features.extract import extract_utterance
        n = 16000 * 2
        rng = np.random.default_rng(0)
        wav = self._make(tmp_path, f"0 {n} s\n", 0.05 * rng.standard_normal(n))
        r = extract_utterance(wav, tmp_path)
        voiced = [f.name for f in FEATURE_ORDER if f.domain in ("voiced", "sonorant")]
        assert all(not np.isfinite(r.features[n_]) for n_ in voiced)
        assert np.isfinite(r.features["RMS"])          # speech-domain still present


# ── Report / QA summary (synthetic) ─────────────────────────────────────────────

class TestReport:
    def _df(self):
        import pandas as pd
        rng = np.random.default_rng(0)
        rows = []
        for spk in range(8):
            base = rng.normal(0, 5)
            for _ in range(4):
                r = {n: rng.normal(0, 1) for n in FEATURE_NAMES}
                r["F0"] = base + rng.normal(0, 0.2)     # speaker-distinctive
                r["speaker_id"] = f"S{spk}"; r["decode_ok"] = True
                rows.append(r)
        return pd.DataFrame(rows)

    def test_coverage_and_ratio(self):
        from timit_features import report
        rep = report.build_report(self._df())
        assert rep["n_speakers"] == 8 and rep["n_utterances"] == 32
        assert rep["coverage"]["F0"] == 1.0
        # F0 (speaker-distinctive) should have a far higher ratio than a noise feature
        assert rep["between_within_speaker_var_ratio"]["F0"] > \
               rep["between_within_speaker_var_ratio"]["jitter"]
