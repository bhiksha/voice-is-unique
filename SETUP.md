# Remote bring-up

How to stand up `voice-is-unique` from scratch on a new machine (laptop, server, or
a cloud GPU box such as Lambda A10), run the tests, and drive the experiments.

The repo is **public** — cloning needs no authentication.

```bash
git clone https://github.com/bhiksha/voice-is-unique.git
cd voice-is-unique
```

## 1. Environments (Miniconda assumed)

```bash
# ── Extractor: 3 isolated conda envs (env names are defined inside each yml) ──
cd extractor
conda env create -f environment.yml             # main extractor env
conda env create -f environment.deepfry.yml     # DeepFry  -> VFI (creak), CPU torch 1.12
conda env create -f environment.deepformants.yml # DeepFormants -> F1-F4

# ── Analysis (TIMIT + Common Voice share the corpus-agnostic core) ──
cd ../timit        && pip install -r requirements.txt
cd ../common-voice && pip install -r requirements.txt

# ── Common Voice masking: Montreal Forced Aligner + English models ──
conda create -y -n aligner -c conda-forge montreal-forced-aligner
conda run -n aligner mfa model download acoustic english_us_arpa
conda run -n aligner mfa model download dictionary english_us_arpa
```

## 2. Smoke test (no data needed)

The acceptance tests use synthetic fixtures, so they pass before any corpus is present
— a good first check on a new box:

```bash
cd timit        && pytest -q     # PR/MI/Fano + Option C recovery, determinism, etc.
cd ../common-voice && pytest -q   # the above (reused) + scaling-sweep invariants
```

## 3. Data (never committed, by design)

- **TIMIT** (LDC-licensed): obtain from the LDC, run the `extractor/` to produce
  `~/data/timit-feats/all_utterances.parquet`, then
  `cd timit && python -m src.run_all --config CONFIG/timit.json`.
- **Common Voice** (CC0/CC-BY): `huggingface-cli login` with your own token, then use
  `common-voice/src/download.py` + `extract.py` to build
  `~/data/commonvoice-feats/all_utterances.parquet`, then
  `cd common-voice && python -m src.run_all --config CONFIG/common_voice.json --pilot`.

Extraction is heavy (~13 CPU-core-seconds/clip: DeepFry + DeepFormants run as per-clip
subprocesses). Parallelize with `--jobs <ncores>` and keep `OMP_NUM_THREADS` low per
worker. On a 4-core laptop ~10k clips ≈ a half day; on a 30-vCPU box ≈ a couple hours.
For large multilingual runs, MFA needs a per-language model (only ~20-30 languages have
one) — restrict to those or use a VAD; and consider running DeepFry/DeepFormants as
persistent GPU services instead of per-clip subprocesses (the dominant speedup lever).

## 4. Run Claude Code on this machine seamlessly

```bash
curl -fsSL https://claude.ai/install.sh | bash      # or: npm i -g @anthropic-ai/claude-code
```

Authenticate headlessly (no browser needed over SSH) — pick one and put it in your
shell profile so every session starts with zero prompts:

```bash
# Subscription: run `claude setup-token` once on a machine you can log in on, then:
echo 'export CLAUDE_CODE_OAUTH_TOKEN=<token>' >> ~/.bashrc
# OR API key (console.anthropic.com):
echo 'export ANTHROPIC_API_KEY=sk-ant-...'   >> ~/.bashrc
```

```bash
cd voice-is-unique && claude
```
