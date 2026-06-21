"""Clean, wandb-free inference wrapper for the vendored DeepFormants PyTorch model.

Exposes F1-F4 estimation for a time window. Run inside the isolated `deepformants`
conda env (it requires torch). The voice-is-unique analysis env calls this as a
subprocess so torch never enters the analysis env.

Provenance and patches: see VENDOR.md.
"""
from __future__ import annotations
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import extract_features as features  # patched for numpy2 / scipy1.17

_ESTIMATOR_PT = os.path.join(_HERE, "models", "LPC_NN_scaledLoss.pt")
_FEATURE_DIM = 350
_DEVICE = torch.device("cpu")   # determinism: CPU only


class _Estimator(torch.nn.Module):
    """MLP from the bundled DeepFormants PyTorch estimator (350 -> F1..F4)."""

    def __init__(self) -> None:
        super().__init__()
        self.Dense1 = torch.nn.Linear(_FEATURE_DIM, 1024)
        self.Dense2 = torch.nn.Linear(1024, 512)
        self.Dense3 = torch.nn.Linear(512, 256)
        self.out = torch.nn.Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(self.Dense1(x))
        x = torch.sigmoid(self.Dense2(x))
        x = torch.sigmoid(self.Dense3(x))
        return self.out(x)


_MODEL: _Estimator | None = None


def _model() -> _Estimator:
    global _MODEL
    if _MODEL is None:
        m = _Estimator().to(_DEVICE)
        m.load_state_dict(torch.load(_ESTIMATOR_PT, map_location=_DEVICE))
        m.eval()
        _MODEL = m
    return _MODEL


def estimate_window(wav_filename: str, begin: float, end: float) -> np.ndarray:
    """Return np.array([F1, F2, F3, F4]) in Hz for the window [begin, end] seconds."""
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".csv")
    try:
        features.create_features(wav_filename, tmp, begin, end)
        row = open(tmp).read().strip().split(",")
        feats = np.array(row[1:], dtype=np.float32).reshape(1, -1)
        if feats.shape[1] != _FEATURE_DIM:
            raise ValueError(f"expected {_FEATURE_DIM} features, got {feats.shape[1]}")
        with torch.no_grad():
            pred = _model()(torch.from_numpy(feats).to(_DEVICE)).cpu().numpy()[0]
        return pred * 1000.0
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def estimate_windows(wav_filename: str, windows):
    """Estimate F1-F4 for many [begin,end] windows; model loaded once.
    Returns an (N,4) array; a row is NaN if its window fails."""
    rows = []
    for (b, e) in windows:
        try:
            rows.append(estimate_window(wav_filename, float(b), float(e)))
        except Exception:
            rows.append(np.full(4, np.nan))
    return np.asarray(rows, dtype=np.float64)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="DeepFormants F1-F4 estimation")
    ap.add_argument("wav")
    ap.add_argument("--windows", help="text file of 'begin end' lines (batch mode)")
    ap.add_argument("--out", help="output CSV for batch mode")
    ap.add_argument("--begin", type=float)
    ap.add_argument("--end", type=float)
    args = ap.parse_args()

    if args.windows:                                   # batch mode
        wins = []
        for ln in open(args.windows):
            ps = ln.split()
            if len(ps) >= 2:
                wins.append((float(ps[0]), float(ps[1])))
        out = estimate_windows(args.wav, wins)
        lines = ["\n".join(",".join(f"{v:.4f}" for v in row) for row in out)]
        text = "\n".join(",".join(f"{v:.4f}" for v in row) for row in out)
        (open(args.out, "w").write(text) if args.out else print(text))
    else:                                              # single window
        f = estimate_window(args.wav, args.begin, args.end)
        print(",".join(f"{v:.4f}" for v in f))
