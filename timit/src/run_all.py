"""One-command driver: PR + summed-MI + Fano on the TIMIT 40-feature table, in
two conditions (pooled and within-sex), reviewer-reproducible.

    python -m src.run_all --config CONFIG/timit.json

Writes aggregate tables (tables/), figures (reports/figs/), provenance
(tables/provenance.json) and the self-contained report (reports/report.md). No
per-utterance or corpus data is written. Deterministic given the config + seeds.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from src import common, pr as PRm, mi as MIm, fano as FAm
from src.vfp_hurdle import fit_hurdle, decompose_var, vfp_utt_composite

NBIN_REF = 5


# ── per-subset computation ──────────────────────────────────────────────────────

def fisher_table(df, cfg, S):
    sp = df[cfg["speaker_key"]].to_numpy()
    speakers = common.sorted_speakers(df, cfg)
    X, names = common.transformed_matrix(df, cfg)
    rows = []
    for j, name in enumerate(names):
        fc = common.fisher_components(X[:, j], sp, speakers)
        rows.append(dict(feature=name, **fc))
    z = vfp_utt_composite(df[cfg["vfp"]["name"]].to_numpy(float))
    rows.append(dict(feature="VFI", **common.fisher_components(z, sp, speakers)))
    rows.sort(key=lambda r: -(r["Fstar"] if np.isfinite(r["Fstar"]) else -1))
    return rows


def pr_block(df, cfg):
    pts = {v: PRm.pr_point(df, cfg, vfp=v) for v in ("C", "B", "none")}
    A = pts["C"]["A"]
    out = {}
    for v, key in [("C", "optionC"), ("B", "optionB"), ("none", "exclude_vfp")]:
        ci, bmean = PRm.pr_bootstrap(pts[v]["A"], cfg, cfg["pr"]["n_boot"],
                                     cfg["pr"]["boot_seed"], vfp=v)
        out[key] = dict(PR=pts[v]["PR"], ci_lo=ci[0], ci_hi=ci[1], boot_mean=bmean,
                        kept=pts[v]["kept"], dropped=pts[v]["dropped"],
                        repaired=pts[v]["repaired"], **pts[v]["crosscheck"])
    nul = PRm.pr_null(df, cfg, 200, cfg["seed"], vfp="C")
    out["null_mean"], out["null_sd"] = float(nul.mean()), float(nul.std())
    order = PRm.fisher_order(df, cfg, A)
    S, names = PRm.sigma_b(A, "C")
    idx = {n: i for i, n in enumerate(names)}
    disp = {"VFI": "VFI(optC)"}
    fo = [disp.get(n, n) for n, _ in order]
    inc = []
    for k in range(1, len(fo) + 1):
        sub = S[np.ix_([idx[fo[t]] for t in range(k)], [idx[fo[t]] for t in range(k)])]
        inc.append((k, fo[k - 1], common.pr_from_cov(sub)))
    out["incremental"] = inc
    return out


def mi_block(df, cfg, S):
    y = df[cfg["speaker_key"]].to_numpy()
    vfp = cfg["vfp"]["name"]
    table = {}
    summed = {}
    for nbin in cfg["mi"]["nbins"]:
        per = {}
        for f in cfg["feature_names"]:
            r = MIm.mi_feature(df[f].to_numpy(float), y, S, nbin, is_vfp=(f == vfp),
                               n_perm=cfg["mi"]["n_perm"], seed=cfg["mi"]["perm_seed"])
            per[f] = r
        table[nbin] = per
        summed[nbin] = float(sum(per[f]["I_corrected"] for f in per))
    pres = MIm.mi_presence(df[vfp].to_numpy(float), y, S, cfg["mi"]["n_perm"],
                           cfg["mi"]["perm_seed"])
    return dict(table=table, summed=summed, vfp_presence=pres)


def fano_block(df, cfg, S):
    D, names, vfp_raw, sp = FAm.make_static_design(df, cfg)
    y = df[cfg["speaker_key"]].to_numpy()
    splits = FAm.cv_splits(y, cfg["fano"]["cv_folds"], cfg["fano"]["cv_seed"])
    res = {}
    for clf in cfg["fano"]["classifiers"]:
        accs, ifs, ixs = [], [], []
        for tr, te in splits:
            r = FAm.run_fold(D, names, vfp_raw, sp, y, tr, te, S, clf)
            accs.append(r["acc"]); ifs.append(r["I_fano"]); ixs.append(r["I_xent"])
        res[clf] = dict(acc_mean=float(np.mean(accs)), acc_sd=float(np.std(accs)),
                        ifano_mean=float(np.mean(ifs)), ifano_sd=float(np.std(ifs)),
                        ixent_mean=float(np.mean(ixs)), ixent_sd=float(np.std(ixs)))
    rng = np.random.default_rng(cfg["seed"])
    tr, te = splits[0]
    rn = FAm.run_fold(D, names, vfp_raw, sp, y, tr, te, S, "logreg",
                      shuffle_y=rng.permutation(y))
    res["null"] = rn
    res["headline_ifano"] = max(res[c]["ifano_mean"] for c in cfg["fano"]["classifiers"])
    res["headline_ixent"] = max(res[c]["ixent_mean"] for c in cfg["fano"]["classifiers"])
    res["state3_impute"] = FAm.state3_impute_counts(df, cfg)
    res["N"] = int(len(y))
    return res


def compute_block(df, cfg):
    S = int(df[cfg["speaker_key"]].nunique())
    return dict(S=S, N=int(len(df)), ceiling=float(np.log2(S)),
                fisher=fisher_table(df, cfg, S), pr=pr_block(df, cfg),
                mi=mi_block(df, cfg, S), fano=fano_block(df, cfg, S))


# ── within-sex combine: Q_within = Σ_g (S_g/S) Q_g ──────────────────────────────

def within_sex(df, cfg):
    sexes = sorted(df[cfg["sex_key"]].unique())
    S = int(df[cfg["speaker_key"]].nunique())
    blocks = {}
    w = {}
    for g in sexes:
        sub = df[df[cfg["sex_key"]] == g]
        blocks[g] = compute_block(sub, cfg)
        w[g] = blocks[g]["S"] / S
    comb = dict(
        PR_optionC=sum(w[g] * blocks[g]["pr"]["optionC"]["PR"] for g in sexes),
        PR_exclude=sum(w[g] * blocks[g]["pr"]["exclude_vfp"]["PR"] for g in sexes),
        mi_summed_ref=sum(w[g] * blocks[g]["mi"]["summed"][NBIN_REF] for g in sexes),
        fano_ifano=sum(w[g] * blocks[g]["fano"]["headline_ifano"] for g in sexes),
        fano_ixent=sum(w[g] * blocks[g]["fano"]["headline_ixent"] for g in sexes),
        ceiling=sum(w[g] * blocks[g]["ceiling"] for g in sexes))
    return dict(by_sex=blocks, weights=w, combined=comb, sexes=sexes)


# ── outputs: tables, provenance, figures, report ────────────────────────────────

def write_tables(pooled, ws, cfg, root: Path):
    t = root / "tables"; t.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pooled["fisher"]).to_csv(t / "fisher_table.csv", index=False)
    for g, blk in ws["by_sex"].items():
        pd.DataFrame(blk["fisher"]).to_csv(t / f"fisher_within_sex_{g}.csv", index=False)
    inc = pd.DataFrame([(k, f, p) for k, f, p in pooled["pr"]["incremental"]],
                       columns=["k", "feature_added", "PR"])
    inc.to_csv(t / "pr_curve.csv", index=False)
    pr_rows = []
    for cond, blk in [("pooled", pooled)] + [(f"within_{g}", ws["by_sex"][g]) for g in ws["sexes"]]:
        for rep in ("optionC", "optionB", "exclude_vfp"):
            d = blk["pr"][rep]
            pr_rows.append(dict(condition=cond, representation=rep, PR=d["PR"],
                                ci_lo=d["ci_lo"], ci_hi=d["ci_hi"],
                                dim_90pct_var=d["dim_90pct_var"],
                                dim_spectral_entropy=d["dim_spectral_entropy"],
                                null_mean=blk["pr"]["null_mean"]))
    pd.DataFrame(pr_rows).to_csv(t / "pr_summary.csv", index=False)
    mi_rows = []
    for cond, blk, Sc in [("pooled", pooled, pooled["S"])] + \
            [(f"within_{g}", ws["by_sex"][g], ws["by_sex"][g]["S"]) for g in ws["sexes"]]:
        for nbin, per in blk["mi"]["table"].items():
            for f, r in per.items():
                mi_rows.append(dict(condition=cond, nbin=nbin, feature=f,
                                    I_raw=r["I_raw"], I_null=r["I_null"],
                                    I_corrected=r["I_corrected"], n_used=r["n"]))
    pd.DataFrame(mi_rows).to_csv(t / "mi_table.csv", index=False)
    fa_rows = []
    for cond, blk in [("pooled", pooled)] + [(f"within_{g}", ws["by_sex"][g]) for g in ws["sexes"]]:
        for clf in cfg["fano"]["classifiers"]:
            d = blk["fano"][clf]
            fa_rows.append(dict(condition=cond, classifier=clf, **d))
        n = blk["fano"]["null"]
        fa_rows.append(dict(condition=cond, classifier="null_shuffled",
                            acc_mean=n["acc"], ifano_mean=n["I_fano"], ixent_mean=n["I_xent"]))
    pd.DataFrame(fa_rows).to_csv(t / "fano_summary.csv", index=False)


def write_provenance(cfg, df, root: Path):
    p = Path(cfg["input_parquet"]).expanduser()
    speakers = df[cfg["speaker_key"]]
    prov = dict(
        input_parquet=str(p), input_sha256_16=sha256(p.read_bytes()).hexdigest()[:16],
        S=int(speakers.nunique()), N=int(len(df)),
        n_i=dict(sorted(pd.Series(speakers).value_counts().value_counts().items())),
        S_by_sex={g: int(df[df[cfg["sex_key"]] == g][cfg["speaker_key"]].nunique())
                  for g in sorted(df[cfg["sex_key"]].unique())},
        config=cfg,
        libs=dict(python=sys.version.split()[0], numpy=np.__version__,
                  pandas=pd.__version__, platform=platform.platform()),
        timestamp=datetime.now(timezone.utc).isoformat())
    (root / "tables" / "provenance.json").write_text(json.dumps(prov, indent=2))
    return prov


def make_figs(pooled, ws, cfg, root: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = root / "reports" / "figs"; fd.mkdir(parents=True, exist_ok=True)

    inc = pooled["pr"]["incremental"]
    plt.figure(figsize=(6, 4))
    plt.plot([k for k, _, _ in inc], [p for _, _, p in inc], "-o", ms=3)
    plt.axhline(pooled["pr"]["optionC"]["PR"], ls="--", c="grey",
                label=f"full PR={pooled['pr']['optionC']['PR']:.2f}")
    plt.xlabel("# features (decreasing F*)"); plt.ylabel("PR"); plt.legend()
    plt.title("Incremental participation ratio (pooled, Option C)")
    plt.tight_layout(); plt.savefig(fd / "pr_incremental.png", dpi=120); plt.close()

    mi5 = pooled["mi"]["table"][NBIN_REF]
    feats = sorted(mi5, key=lambda f: -mi5[f]["I_corrected"])
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(feats)), [mi5[f]["I_corrected"] for f in feats])
    plt.xticks(range(len(feats)), feats, rotation=90, fontsize=6)
    plt.ylabel("I_corrected (bits)"); plt.title(f"Per-feature MI (pooled, Nbin={NBIN_REF})")
    plt.tight_layout(); plt.savefig(fd / "mi_per_feature.png", dpi=120); plt.close()

    # bracket figure
    fano = pooled["fano"]["headline_ixent"]
    mi_sum = pooled["mi"]["summed"][NBIN_REF]
    ceil = pooled["ceiling"]
    plt.figure(figsize=(6, 3))
    plt.barh(["Fano lower", "ceiling log2 S", "summed MI (upper)"],
             [fano, ceil, mi_sum], color=["#4477aa", "#888888", "#cc6677"])
    plt.axvline(pooled["pr"]["optionC"]["PR"], ls=":", c="green",
                label=f"PR d_eff={pooled['pr']['optionC']['PR']:.1f}")
    plt.xlabel("bits"); plt.legend(); plt.title("Joint speaker-information bracket (pooled)")
    plt.tight_layout(); plt.savefig(fd / "bracket.png", dpi=120); plt.close()


def write_report(pooled, ws, cfg, prov, root: Path):
    from src.report import build_report
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "report.md").write_text(build_report(pooled, ws, cfg, prov))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="CONFIG/timit.json")
    ap.add_argument("--out", default=".")
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    np.random.seed(cfg["seed"])
    df = common.load_table(cfg)
    root = Path(args.out)

    print("[1/4] pooled ...", flush=True)
    pooled = compute_block(df, cfg)
    print("[2/4] within-sex ...", flush=True)
    ws = within_sex(df, cfg)
    print("[3/4] tables + provenance + figures ...", flush=True)
    write_tables(pooled, ws, cfg, root)
    prov = write_provenance(cfg, df, root)
    make_figs(pooled, ws, cfg, root)
    print("[4/4] report ...", flush=True)
    write_report(pooled, ws, cfg, prov, root)
    print("DONE. PR(optC)=%.3f  Fano(ixent)=%.3f  summedMI(Nbin5)=%.3f  ceiling=%.3f"
          % (pooled["pr"]["optionC"]["PR"], pooled["fano"]["headline_ixent"],
             pooled["mi"]["summed"][NBIN_REF], pooled["ceiling"]), flush=True)


if __name__ == "__main__":
    main()
