# Running the experiments on PSC (Bridges-2)

Step-by-step for porting this package to **PSC Bridges-2**, where the 128-core nodes turn the
multi-day laptop runs into a few hours. Covers **both** turnkey experiments:
- **TIMIT** — 40-feature extraction + PR/MI/Fano analysis (`timit/run_timit_psc.slurm`).
- **Common Voice** — download the exact 8k-study clips → MFA-align → 40 features → **Dr.VOT** (merged
  into the VOT feature) → speaker-count scaling analysis (balanced N=1000..8000) + female-only +
  male-only (`common-voice/run_commonvoice_psc.slurm`).

Placeholders used below: `<USER>` = your PSC username, `<GRANT>` = your allocation ID.

> Note: command details (partitions, `interact`/`sbatch` flags) follow standard Bridges-2 usage —
> verify against the current user guide: https://www.psc.edu/resources/bridges-2/user-guide/

---

## (a) Log in + get a terminal

**Option 1 — SSH (from your laptop):**
```bash
ssh <USER>@bridges2.psc.edu
```
Authenticate with your **PSC password** (set in the ACCESS/PSC portal) and **DUO 2-factor**. You land
on a *login node* — fine for editing, git, and small setup; **never run heavy compute here**.

**Option 2 — Web (no SSH):** open **https://ondemand.bridges2.psc.edu**, log in with the same
credentials → *Clusters ▸ Bridges-2 Shell Access* for a browser terminal. (OnDemand also has a file
browser and interactive desktops.)

Confirm your allocation and storage path:
```bash
projects          # lists your <GRANT> id(s) and the Ocean path
```

## (b) Set up disk

Two filesystems matter:
- **`$HOME`** = `/jet/home/<USER>` — only **25 GB**. Too small for data *or* conda envs. Don't put bulk here.
- **Ocean** = `/ocean/projects/<GRANT>/<USER>` — large project space (TBs). **Put everything here.**

```bash
export OCEAN=/ocean/projects/<GRANT>/$USER     # also add this line to ~/.bashrc
mkdir -p $OCEAN/vu/{repo,data,envs,pkgs}
cd $OCEAN/vu
```

**Critical:** conda envs are several GB (torch) and will blow the 25 GB `$HOME` quota, so point conda
at Ocean:
```bash
module load anaconda3
conda config --add envs_dirs $OCEAN/vu/envs
conda config --add pkgs_dirs $OCEAN/vu/pkgs
```

## (c) Check out the package

```bash
cd $OCEAN/vu/repo
git clone https://github.com/bhiksha/voice-is-unique.git
cd voice-is-unique
export REPO=$PWD
```
This brings the whole package: `common-voice/` (scripts + frozen manifests) and `extractor/` (the
40-feature `timit_features` code + vendored DeepFormants/DeepFry + env specs).

## (d) Set up environments + run download → features

**1. Create the conda envs.** Easiest — use the committed helper (builds all 4 envs into Ocean +
downloads the MFA models), on a login node:
```bash
GRANT=<GRANT> bash setup_psc.sh
```
<details><summary>…or do it by hand (equivalent):</summary>

```bash
conda env create -p $OCEAN/vu/envs/voice-is-unique -f extractor/environment.yml
conda env create -p $OCEAN/vu/envs/deepformants  -f extractor/environment.deepformants.yml
conda env create -p $OCEAN/vu/envs/deepfry       -f extractor/environment.deepfry.yml
# MFA (its env spec isn't in the repo) + the acoustic/dictionary models:
conda create -p $OCEAN/vu/envs/aligner -c conda-forge montreal-forced-aligner -y
conda run -p $OCEAN/vu/envs/aligner mfa model download acoustic   english_us_arpa
conda run -p $OCEAN/vu/envs/aligner mfa model download dictionary english_us_arpa
```
</details>

**2. HuggingFace access** (Common Voice is gated): accept the dataset terms for
`fsicoli/common_voice_22_0` (and `_21_0`, `_17_0`) on huggingface.co while logged in, then:
```bash
conda run -p $OCEAN/vu/envs/voice-is-unique huggingface-cli login   # paste your HF token
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

**3. (no path fix needed)** The scripts now resolve the extractor relative to the repo, so they run
unchanged on PSC. (Override with `EXTRACTOR_SRC`/`CV_PKG` env vars only if you relocate the code.)

**4. Smoke-test interactively** (grab a shared slice of a node, ~15 min):
```bash
interact -p RM-shared -n 8 -t 1:00:00          # 8 cores, interactive shell on a compute node
cd $REPO/common-voice
D=$OCEAN/vu/data/cv_study
# download + decode + MFA-align a few speakers:
conda run -p $OCEAN/vu/envs/voice-is-unique python scripts/reproduce_corpus.py \
   --manifest-dir manifest/study8k --out $D --align --skip-classify --limit 5
# extract 40 features for those:
conda run -p $OCEAN/vu/envs/voice-is-unique python scripts/extract_feats_8k.py \
   --use-manifest manifest/feats_manifest_8000.tsv --corpus $D --wavroot $D/wavs \
   --out ${D}-feats --jobs 8 --limit 50
exit
```
If that produces `${D}-feats/.../*.json`, the pipeline works.

**5. Full runs as batch jobs.** The repo ships two end-to-end SLURM scripts. Set `GRANT` (edit the
`GRANT=<EDIT_ME>` line or `export GRANT=<GRANT>`), then submit:

**Common Voice — the entire experiment** (download exact 8k clips → MFA-align → 40 features →
Dr.VOT merged into the VOT feature → verify → scaling analysis N=1000..8000 + female-only +
male-only):
```bash
cd $REPO/common-voice
export GRANT=<GRANT>
sbatch run_commonvoice_psc.slurm
squeue -u $USER ; tail -f cv_full.*.out
```
Outputs: features `$OCEAN/vu/data/cv_study-feats/all_utterances.parquet`; VOT
`…/cv_vot/vot_pilot.tsv`; analysis `…/cv-analysis/tables/scaling_{pooled,within_sex,female_only,male_only}.csv`
+ `reports/figs/scaling_all.png`.

**TIMIT — the entire experiment** (40-feature extraction → PR/MI/Fano + report). TIMIT is
LDC-licensed, so bring your own copy and point `TIMIT_ROOT` at the dir with `TRAIN/` and `TEST/`:
```bash
cd $REPO/timit
export GRANT=<GRANT> TIMIT_ROOT=/ocean/projects/<GRANT>/$USER/timit/TIMIT
sbatch run_timit_psc.slurm
tail -f timit.*.out
```
Outputs: features `$OCEAN/vu/data/timit-feats/`; analysis `$OCEAN/vu/data/timit-analysis/`.

Both jobs are **resumable** — re-`sbatch` continues (download/align/features/VOT-shards/JSONs all
skip completed work).
On a 128-core RM node the 40-feature extraction drops from ~2.7 days (laptop) to **~2–3 hours**;
MFA alignment is similarly parallel.

---

## Honest caveats
- **Setup, not the run, is the work** — recreating 4 conda envs + MFA models is the bulk of the
  effort; budget ~an hour for step (d.1).
- I can't test PSC directly, so verify command/partition details against the Bridges-2 user guide
  (above). The structure here is standard Bridges-2.
- `reproduce_corpus.py --align` writes many small files — Ocean handles that fine (unlike the laptop's
  `~/data` 9p/`/mnt/c` mount).
- DeepFormants/DeepFry charge Service Units per node-hour; the ~2–3 h RM run is modest.

## Mapping to the laptop layout (reference)
| laptop | PSC |
|--------|-----|
| `~/claude/vu-pkg` (repo) | `$OCEAN/vu/repo/voice-is-unique` |
| `~/miniconda3/envs/<env>` | `$OCEAN/vu/envs/<env>` |
| `~/data/commonvoice*`, `~/cv_align` (working data) | `$OCEAN/vu/data/...` (set via `--out/--corpus/--wavroot`) |
| `--jobs 6` (laptop, 4 cores) | `--jobs 120` (RM node, 128 cores) |

## Optional: Dr.VOT on a GPU node
The Dr.VOT VOT job (`common-voice/scripts/drvot_vot.py`, 422k voiceless stops) is the slow one on the
laptop (~20 h, CPU). Dr.VOT has a CUDA path, so a Bridges-2 **GPU** partition (`-p GPU-shared
--gres=gpu:v100-16:1`) would finish it in well under an hour — ask and a GPU SLURM script can be added.

## Turnkey: clone → edit GRANT → sbatch
Everything is committed, so PSC use is minimal:
1. `setup_psc.sh` (repo root) — builds the 5 named conda envs (voice-is-unique, deepformants,
   deepfry, drvot, aligner) into Ocean + `pip install -e extractor` + MFA models;
2. `common-voice/run_commonvoice_psc.slurm` — the full CV experiment (download→features+VOT→analysis);
3. `timit/run_timit_psc.slurm` — the full TIMIT experiment (features→analysis);
4. scripts resolve code/Dr.VOT paths relative to the repo (no symlink shim).

End to end on Bridges-2:
```bash
git clone https://github.com/bhiksha/voice-is-unique.git && cd voice-is-unique
GRANT=<GRANT> bash setup_psc.sh
conda run -n voice-is-unique huggingface-cli login          # Common Voice is gated
cd common-voice && export GRANT=<GRANT> && sbatch run_commonvoice_psc.slurm
# TIMIT (bring your own LDC corpus):
cd ../timit && export GRANT=<GRANT> TIMIT_ROOT=<path-to-TIMIT> && sbatch run_timit_psc.slurm
```

## Verified vs. not
- The **analysis driver** (`run_scaling.py`) and the per-stage scripts were exercised locally; the
  `--shards/--max-parallel` Dr.VOT runner and feature/embedding steps ran on the laptop.
- I have **not** run the SLURM scripts on a live PSC node — verify partition/walltime/flags against
  the Bridges-2 user guide. The structure (named envs in Ocean, `RM` partition, `conda run -n`) is
  standard Bridges-2.
- Dr.VOT shard counts in `run_commonvoice_psc.slurm` (`--shards 64 --max-parallel 32`) are sized for
  the 256 GB / 128-core RM node; tune if needed. On a GPU node Dr.VOT would be far faster (CUDA path).
