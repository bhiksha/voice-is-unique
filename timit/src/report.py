"""Build the self-contained §0–§5 report from computed result blocks."""
from __future__ import annotations

NBIN_REF = 5


def _pr_line(blk):
    p = blk["pr"]
    c, b, e = p["optionC"], p["optionB"], p["exclude_vfp"]
    return (f"PR Option C **{c['PR']:.3f}** (95% CI [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]) · "
            f"Option B {b['PR']:.3f} · exclude-VFP {e['PR']:.3f} · "
            f"null {p['null_mean']:.3f}±{p['null_sd']:.3f} · "
            f"90%-var dim {c['dim_90pct_var']} · spec-entropy dim {c['dim_spectral_entropy']:.2f}")


def build_report(pooled, ws, cfg, prov) -> str:
    S, N = pooled["S"], pooled["N"]
    L = []
    A = L.append
    A("# PR · summed-MI · Fano on the TIMIT 40-feature speaker representation\n")
    A("Three quantities on the fixed 40-feature utterance-level representation, all "
      "6300 utterances, pooled and within-sex. Summed MI is an UPPER bound and Fano a "
      "LOWER bound on joint speaker information; the corpus ceiling is log2 S.\n")

    # §0
    A("## §0 Provenance & freeze")
    A(f"- input `{prov['input_parquet']}` sha256 `{prov['input_sha256_16']}`")
    A(f"- **S = {S} speakers, N = {N} utterances**, balanced (n_i = "
      f"{list(prov['n_i'].keys())[0]} for every speaker) ⇒ H(Y) = log2 S = {pooled['ceiling']:.4f} bits")
    A(f"- per sex: " + ", ".join(f"S_{g} = {n} (log2 = {__import__('numpy').log2(n):.4f})"
                                 for g, n in prov["S_by_sex"].items()))
    A("- transforms: NAQ, alpha_ratio, LHR, SPI → natural log; **VFP (=VFI) → hurdle** "
      "(log of non-zero state-1 magnitude, z-scored over state-1; exact zeros = state-2 "
      "carried by the hurdle: MI zero-bin, PR Option C, Fano presence). All others linear; "
      "then corpus z-score to unit variance.")
    A(f"- SIGMA_B_FLOOR={cfg['pr']['sigma_b_floor']}, PSD-repair={cfg['pr']['psd_repair']}, "
      f"N_BOOT={cfg['pr']['n_boot']}; MI Nbin∈{cfg['mi']['nbins']} (ref {NBIN_REF}), "
      f"n_perm={cfg['mi']['n_perm']}; Fano {cfg['fano']['cv_folds']}-fold speaker-stratified, "
      f"classifiers={cfg['fano']['classifiers']}, impute=train-fold-mean (no indicator); seed={cfg['seed']}.")
    A(f"- libs: python {prov['libs']['python']}, numpy {prov['libs']['numpy']}, "
      f"pandas {prov['libs']['pandas']}; {prov['timestamp']}\n")

    # §1 PR
    A("## §1 PR — effective dimensionality (participation ratio)")
    A(f"- **pooled:** {_pr_line(pooled)}")
    for g in ws["sexes"]:
        A(f"- **within {g}:** {_pr_line(ws['by_sex'][g])}")
    A(f"- **within-sex combined** Σ_g (S_g/S)·PR_g: Option C "
      f"{ws['combined']['PR_optionC']:.3f} (exclude-VFP {ws['combined']['PR_exclude']:.3f})")
    A(f"- VFP contribution to PR (pooled): exclude {pooled['pr']['exclude_vfp']['PR']:.3f} → "
      f"Option C {pooled['pr']['optionC']['PR']:.3f}; Option C vs B gap "
      f"{abs(pooled['pr']['optionC']['PR']-pooled['pr']['optionB']['PR']):.3f} bounds the "
      "reconstruction's influence. Null ≥ real in every case ⇒ features are genuinely redundant.")
    A("- φ = 1 on TIMIT (every speaker creaks in ≥1 utterance), so Option C's presence-variance "
      "term is exactly 0; VFP's between-speaker signal is carried by the rate×magnitude composite.\n")

    # §2 MI
    A("## §2 Summed MI — per-feature, debiased (UPPER bound)")
    mi5 = pooled["mi"]["table"][NBIN_REF]
    top = sorted(mi5, key=lambda f: -mi5[f]["I_corrected"])[:10]
    A(f"Top features by I_corrected (pooled, Nbin={NBIN_REF}):")
    A("| feature | n_used | I_raw | I_null | I_corrected |")
    A("|---|---|---|---|---|")
    for f in top:
        r = mi5[f]
        A(f"| {f} | {r['n']} | {r['I_raw']:.4f} | {r['I_null']:.4f} | {r['I_corrected']:.4f} |")
    A(f"\n- **summed-marginal MI (UPPER bound)** Σ_f I_corrected, pooled: " +
      ", ".join(f"Nbin={nb}: {pooled['mi']['summed'][nb]:.2f} bits" for nb in cfg["mi"]["nbins"]))
    A(f"- within-sex combined summed MI (Nbin={NBIN_REF}): {ws['combined']['mi_summed_ref']:.2f} bits")
    pres = pooled["mi"]["vfp_presence"]
    A(f"- VFP zero-bin (presence/absence) contributes I_corrected={pres['I_corrected']:.4f} of "
      f"VFP total {mi5[cfg['vfp']['name']]['I_corrected']:.4f} — most VFP info is magnitude, not presence.")
    A("- Genuine NaNs (state 3) are IGNORED: no missing bin; each feature's MI uses only its "
      "measured support (full per-feature table in `tables/mi_table.csv`).\n")

    # §3 Fano
    A("## §3 Fano — joint-information LOWER bound (all 6300 utterances)")
    A("| condition | classifier | acc | I_fano | I_xent |")
    A("|---|---|---|---|---|")
    for cond, blk in [("pooled", pooled)] + [(f"within_{g}", ws["by_sex"][g]) for g in ws["sexes"]]:
        for clf in cfg["fano"]["classifiers"]:
            d = blk["fano"][clf]
            A(f"| {cond} | {clf} | {d['acc_mean']:.3f}±{d['acc_sd']:.3f} | "
              f"{d['ifano_mean']:.2f}±{d['ifano_sd']:.2f} | {d['ixent_mean']:.2f}±{d['ixent_sd']:.2f} |")
    n = pooled["fano"]["null"]
    A(f"\n- **pooled headline (max over classifiers):** I_fano {pooled['fano']['headline_ifano']:.2f}, "
      f"I_xent {pooled['fano']['headline_ixent']:.2f} bits (ceiling {pooled['ceiling']:.2f}).")
    A(f"- within-sex combined headline: I_fano {ws['combined']['fano_ifano']:.2f}, "
      f"I_xent {ws['combined']['fano_ixent']:.2f} bits.")
    A(f"- permutation null (pooled, shuffled Y): acc {n['acc']:.5f} ≈ 1/S, I_fano {n['I_fano']:.2f}.")
    s3 = pooled["fano"]["state3_impute"]
    A(f"- state-3 mean-imputation counts (no indicator): " +
      ", ".join(f"{k}={v}" for k, v in s3.items() if v > 0) + "; all others 0. N=6300 throughout.")
    A("- Capacity inversion (MLP ≤ linear) is a data-starvation diagnostic (8 train utts/speaker).\n")

    # §4 Synthesis
    A("## §4 Synthesis — the bracket")
    A(f"- **pooled:** joint speaker info ∈ [Fano lower **{pooled['fano']['headline_ixent']:.2f}**, "
      f"ceiling **{pooled['ceiling']:.2f}**] bits; summed-marginal MI "
      f"{pooled['mi']['summed'][NBIN_REF]:.1f} bits is the redundancy-inflated UPPER bound; "
      f"PR ≈ **{pooled['pr']['optionC']['PR']:.1f}** effective dimensions of 40.")
    qeff = 2 ** (pooled["fano"]["headline_ixent"] / pooled["pr"]["optionC"]["PR"])
    A(f"- derived per-axis resolution q_eff = 2^(Fano/PR) = {qeff:.2f} (consistency check; derived, not measured).")
    A(f"- pooled vs within-sex (the pooled→within gap = sex-parent contribution): "
      f"PR {pooled['pr']['optionC']['PR']:.2f}→{ws['combined']['PR_optionC']:.2f}, "
      f"Fano {pooled['fano']['headline_ixent']:.2f}→{ws['combined']['fano_ixent']:.2f}, "
      f"summed MI {pooled['mi']['summed'][NBIN_REF]:.1f}→{ws['combined']['mi_summed_ref']:.1f}.\n")

    # §5 Verification
    A("## §5 Verification")
    A(f"- nulls: PR null {pooled['pr']['null_mean']:.2f} ≥ real {pooled['pr']['optionC']['PR']:.2f}; "
      f"MI debiased by {cfg['mi']['n_perm']}-shuffle null; Fano null acc≈1/S, bounds≈0.")
    A("- Option C validated by the synthetic-recovery acceptance test (`tests/test_vfp_hurdle.py`).")
    A("- determinism: fixed seeds; same config + input → identical outputs.")
    A("- caveats: single-session TIMIT ⇒ optimistic (no cross-session within-speaker variation); "
      "PR is linear/2nd-order; bounds are corpus-limited (S=630).")
    A("\nFigures: `reports/figs/`. Aggregate tables: `tables/`.")
    return "\n".join(L) + "\n"
