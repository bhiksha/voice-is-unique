# Vendored DeepFormants (formant estimator for F1–F4)

This is a pinned, patched subset of **MLSpeech/DeepFormants**, used for the F1–F4
part of the hybrid formant estimator (see project `DECISIONS.md` #D). F5 and all
bandwidths come from order-20 Burg LPC in the main pipeline, not from here.

## Source / provenance
- Upstream: https://github.com/MLSpeech/DeepFormants
- Pinned commit: **53e2541f7bbca5c78913a82f066288030e3b45d2**
- Model SHA-256 (verify after any update):
  - `models/LPC_NN_scaledLoss.pt` (estimator MLP): `1be6ba5a7a833fdd8e47327ec15343bd5be44a4b7dfd53b863c269054ce1807f`
  - `models/LPC_RNN.pt` (tracker LSTM): `b799e1aa06ada376a665c9b5cc3f81b671c284616aa4d8be17468c94ab2b1856`

## What was included (and why the rest was dropped)
Included: `extract_features.py`, `levinson_lpc.py` (pure-Python LPC; replaces the
abandoned `scikits.talkbox`), `helpers/`, the two **PyTorch** model files, the
upstream `LICENSE`, and `df_infer.py` (our clean inference wrapper).

Dropped: all Lua-Torch files (`*.lua`, `estimation_model.dat`), `ArspecExtract.py`
(the only `scikits.talkbox` user), and `pytorchFormants/Tracker/LPC_model_apply.py`
(imports `wandb` at module load — a training artifact; we reimplement the LSTM
forward pass in a wandb-free wrapper when the tracker is wired in).

## Patches applied (for numpy 2.4 / scipy 1.17)
In `extract_features.py`:
1. `np.fromstring(dstr, np.int16)` → `np.frombuffer(dstr, np.int16)`
   (numpy 2.0 removed binary `fromstring`).
2. `from scipy.signal import lfilter, hamming` →
   `from scipy.signal import lfilter` + `from scipy.signal.windows import hamming`
   (scipy ≥1.13 removed `scipy.signal.hamming`).

## Runtime
Run inside the isolated **`deepformants`** conda env (has torch); the analysis env
stays torch-free and calls this as a subprocess. Also needs the system `sox` binary.

Feature pipeline: 350-D vector = periodogram (`specPS`, pitch 50) + AR spectra for
LPC orders 8..17 → MLP (350→1024→512→256→4) → F1–F4 (×1000 Hz). CPU-only,
deterministic (verified byte-identical across runs).

Usage:
    conda run -n deepformants python df_infer.py <wav> <begin_s> <end_s>
"""
