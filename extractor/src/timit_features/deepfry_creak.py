"""VFI (#11) via the vendored DeepFry creak detector (arXiv 2203.17019).

Runs DeepFry in its isolated `deepfry` conda env as a subprocess (writing a temp
PCM WAV so it never decodes SPHERE), parses the predicted creak intervals from the
output TextGrid, and returns a per-frame array: 1.0 for voiced frames whose center
falls in a creak interval, 0.0 for other voiced frames, NaN elsewhere. The
frame→utterance MEAN over voiced frames then gives the proportion of creaky voiced
frames (= Vocal Fry Index).
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from timit_features.config import Config

_CONDA = os.path.expanduser(os.environ.get("CV_CONDA", "~/miniconda3/bin/conda"))
_DF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                       "third_party", "deepfry")
_MODELS = {
    "paper": "CREAK-220lr_0.001_decay_21_input_size_512_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_.pth",
    "both_datasets": "CREAK-74lr_0.001_decay_38_input_size_128_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_logtxt_ff1_.pth",
}

_INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = \"([^\"]*)\"")


def _parse_creak_intervals(tg_path: str, tier: str, mark: str):
    """Return [(xmin, xmax), ...] for intervals marked `mark` in tier `tier`."""
    text = open(tg_path, encoding="utf-8").read()
    # isolate the target tier block (from its name to the next tier name or EOF)
    m = re.search(rf'name = "{re.escape(tier)}".*?(?=item \[|\Z)', text, re.DOTALL)
    block = m.group(0) if m else text
    return [(float(a), float(b)) for a, b, t in _INTERVAL_RE.findall(block)
            if t.strip() == mark]


def _deepfry_intervals(signal, sr, model_path, retries: int = 1):
    """Run DeepFry (with one retry to absorb transient parallel-run failures).
    Returns the list of creak intervals, or None if DeepFry produced no result."""
    for _ in range(retries + 1):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "test"))
            sf.write(os.path.join(d, "test", "u.wav"),
                     np.asarray(signal, dtype=np.float32), sr, subtype="PCM_16")
            out_dir = os.path.join(d, "out"); os.makedirs(out_dir)
            try:
                subprocess.run(
                    [_CONDA, "run", "-n", "deepfry", "python", "run.py",
                     "--data_dir", d, "--custom", "--output_dir", out_dir,
                     "--model_name", model_path],
                    cwd=_DF_DIR, check=True, capture_output=True, timeout=600)
                tgs = glob.glob(os.path.join(out_dir, "*.TextGrid"))
                if tgs:
                    return _parse_creak_intervals(tgs[0], "pred-creaky", "c")
            except Exception:
                pass
    return None


def compute(signal: np.ndarray, frames, config: Config) -> np.ndarray:
    """Per-frame VFI indicator. Voiced frames get 1 if their centre is in a
    DeepFry creak interval, else 0; non-voiced frames are NaN.

    Policy (DECISIONS): a *voiced* frame with no creak is 0 fry. If DeepFry yields
    no result at all (after one retry — e.g. its 1-D-output IndexError on certain
    ~4.0 s utterances), the utterance is still voiced with no detected fry, so its
    voiced frames are set to 0 (not NaN). VFI is NaN only when there are no voiced
    frames. (Transient parallel-run failures are absorbed by the retry.)"""
    sr = config.framing.sample_rate_expected
    out = np.full(frames.n_frames, np.nan)
    model_path = os.path.join(_DF_DIR, "models", _MODELS[config.creak.model])

    creak = _deepfry_intervals(signal, sr, model_path)
    out[frames.voiced] = 0.0                            # voiced + no fry → 0 (incl. no-result)
    if creak:
        centers_t = frames.center_samples / sr
        is_creak = np.zeros(frames.n_frames, dtype=bool)
        for (a, b) in creak:
            is_creak |= (centers_t >= a) & (centers_t < b)
        out[frames.voiced & is_creak] = 1.0
    return out
