"""Speaker-count scaling sweep (§D/§E).

Recompute PR, summed-MI and Fano on NESTED, balanced speaker subsets of increasing
size and plot each against N with the corpus ceiling log2 N. Invariants enforced:
  - nested subsets: the size-n speaker set ⊂ the size-(n+step) set (one seeded
    ordering per sex; size-n = first n of each sex);
  - fixed clips per speaker (m) at every size — only N varies;
  - fixed standardization: transform/z-score stats from the FULL set, reused at
    every size (PR is scale-invariant and MI quantile-invariant, so this is exact;
    full-set stats are exposed so the standardization-fixed check is explicit).

Reuses the corpus-agnostic analysis core (pr/mi/fano/common).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import common, pr as PRm, mi as MIm, fano as FAm


# ── full-set standardization stats (computed once, reused at every size) ─────────

def fullset_stats(df_full, cfg) -> dict:
    """Per-feature transform + (mean, std) over the FULL set; VFP log-magnitude
    (mean, std) over full-set state-1. Reused for every subset (only N varies)."""
    stats = {}
    for f in cfg["feature_names"]:
        if f == cfg["vfp"]["name"]:
            continue
        v = df_full[f].to_numpy(float)
        if common.transform_of(f, cfg) == "log":
            v = np.log(v)
        m = np.isfinite(v)
        stats[f] = (float(v[m].mean()), float(v[m].std()))
    val = df_full[cfg["vfp"]["name"]].to_numpy(float)
    st = common.state_codes(val)
    lm = np.log(val[st == 1])
    stats["__vfp_mag__"] = (float(lm.mean()), float(lm.std()))
    return stats


# ── nested, balanced, fixed-m subset selection ──────────────────────────────────

def nested_order(df_full, cfg) -> dict:
    """One seeded ordering of speakers per sex; size-n subset = first n of each."""
    rng = np.random.default_rng(cfg["scaling"]["nested_order_seed"])
    order = {}
    for g in sorted(df_full[cfg["sex_key"]].unique()):
        spk = np.array(sorted(df_full[df_full[cfg["sex_key"]] == g][cfg["speaker_key"]].unique()))
        order[g] = spk[rng.permutation(len(spk))]
    return order


def subset(df_full, cfg, n_per_sex, order, m_clips) -> pd.DataFrame:
    """Nested balanced subset: first n_per_sex speakers/sex, first m clips/speaker."""
    keep_spk = np.concatenate([order[g][:n_per_sex] for g in order])
    sub = df_full[df_full[cfg["speaker_key"]].isin(keep_spk)].copy()
    if "clip_id" in sub.columns:
        sub = sub.sort_values("clip_id")
    sub = sub.groupby(cfg["speaker_key"], group_keys=False).head(m_clips)
    return sub.reset_index(drop=True)


# ── per-size measurements (pooled + within-sex) ─────────────────────────────────

def _summed_mi(df, cfg, S):
    y = df[cfg["speaker_key"]].to_numpy()
    vfp = cfg["vfp"]["name"]
    tot = 0.0
    for f in cfg["feature_names"]:
        r = MIm.mi_feature(df[f].to_numpy(float), y, S, cfg["mi"]["ref_nbin"],
                           is_vfp=(f == vfp), n_perm=cfg["mi"]["n_perm"],
                           seed=cfg["mi"]["perm_seed"])
        tot += r["I_corrected"]
    return tot


def _fano_headline(df, cfg, S):
    D, names, vfp_raw, sp = FAm.make_static_design(df, cfg)
    y = df[cfg["speaker_key"]].to_numpy()
    splits = FAm.cv_splits(y, cfg["fano"]["cv_folds"], cfg["fano"]["cv_seed"])
    best = -np.inf
    for clf in cfg["fano"]["classifiers"]:
        ix = [FAm.run_fold(D, names, vfp_raw, sp, y, tr, te, S, clf)["I_xent"]
              for tr, te in splits]
        best = max(best, float(np.mean(ix)))
    return best


def measure(df, cfg) -> dict:
    S = int(df[cfg["speaker_key"]].nunique())
    pr = PRm.pr_point(df, cfg, vfp="C")["PR"]
    return dict(S=S, ceiling=float(np.log2(S)), PR=pr,
                summed_mi=_summed_mi(df, cfg, S), fano=_fano_headline(df, cfg, S))


def measure_within_sex(df, cfg) -> dict:
    S = int(df[cfg["speaker_key"]].nunique())
    sexes = sorted(df[cfg["sex_key"]].unique())
    parts, w = {}, {}
    for g in sexes:
        sub = df[df[cfg["sex_key"]] == g]
        parts[g] = measure(sub, cfg)
        w[g] = parts[g]["S"] / S
    return dict(
        S=S, ceiling=sum(w[g] * parts[g]["ceiling"] for g in sexes),
        PR=sum(w[g] * parts[g]["PR"] for g in sexes),
        summed_mi=sum(w[g] * parts[g]["summed_mi"] for g in sexes),
        fano=sum(w[g] * parts[g]["fano"] for g in sexes))


def sweep(df_full, cfg, grid, m_clips, condition="pooled") -> pd.DataFrame:
    order = nested_order(df_full, cfg)
    fn = measure if condition == "pooled" else measure_within_sex
    rows = []
    for n in grid:
        sub = subset(df_full, cfg, n, order, m_clips)
        r = fn(sub, cfg)
        r["n_per_sex"] = n
        r["condition"] = condition
        rows.append(r)
    return pd.DataFrame(rows)


def scaling_figure(df_pooled, df_within, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, d, title in [(axes[0], df_pooled, "pooled"), (axes[1], df_within, "within-sex")]:
        N = d["S"].to_numpy()
        ax.plot(N, d["ceiling"], "k--", label="ceiling log2 N")
        ax.plot(N, d["fano"], "-o", label="Fano lower (bits)")
        ax.plot(N, d["summed_mi"], "-s", label="summed MI (upper, bits)")
        ax.plot(N, d["PR"], "-^", label="PR (d_eff)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("speakers N (log2)"); ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle("Speaker-count scaling: PR / Fano / summed-MI vs N")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
