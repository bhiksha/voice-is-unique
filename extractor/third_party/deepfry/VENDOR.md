# Vendored DeepFry (creak detector for VFI, feature #11)

Pinned subset of **bronichern/DeepFry** — the creaky-voice detector used to compute
the Vocal Fry Index (proportion of voiced frames in creak), per the author's
instruction and paper ref [62].

## Source / provenance
- Upstream: https://github.com/bronichern/DeepFry
- Paper: Chernyak et al., "DeepFry: Identifying Vocal Fry Using Deep Neural
  Networks", Interspeech 2022 (arXiv:2203.17019).
- Pinned commit: **5ddf4a45b9aee68d62a880a69759cfa08fe0097a**
- Model SHA-256:
  - `models/CREAK-220...pth` (paper model, default): `297f81d731813a7aad464368744f29ce8dfca4377bab1a7bc22882d65f47278f`
  - `models/CREAK-74...pth`  (both datasets): `b9138bacde63f3b50044d9555a81e5f628040bbbcbefc7b8ac419a649cd0e5e5`

## Runtime
Runs in the isolated **`deepfry`** conda env (Python 3.8 / torch 1.12 CPU;
`mkl=2024.0.0` pinned because torch 1.12 needs `iJIT_NotifyEvent`, removed in
mkl≥2025). Invoked as a subprocess by `timit_features.deepfry_creak`, so the
torch-1.12 / numpy-1.22 stack never touches the analysis env.

CPU-only and deterministic (run.py fixes all seeds + `cudnn.deterministic`).
Predicts creak per **5 ms** frame; `--custom` mode writes an output TextGrid with
a `pred-creaky` tier whose intervals are marked `"c"`.

Usage (custom dataset, no input annotations):
    conda run -n deepfry python run.py --data_dir <dir> --custom \
        --output_dir <out> --model_name <abs path to a models/*.pth>
where <dir>/test/*.wav are standard-PCM WAVs.

## Included / dropped
Included: run.py, models.py, multi_train.py, dataset.py, utils.py, data/, the two
pretrained models, LICENSE. Dropped: training scripts/notebooks, the allstar demo
data, HuBERT-encoder path (not needed; bundled models are the custom encoder).
