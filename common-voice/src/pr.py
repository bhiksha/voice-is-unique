"""PR — participation ratio of the between-speaker correlation matrix (§2).

Σ_b is the covariance of per-speaker mean vectors, built with PAIRWISE-complete
observations (each feature pair uses speakers measurable on both — no speaker is
globally deleted for one missing feature). A degenerate screen drops features with
between-speaker SD < SIGMA_B_FLOOR; nearest-PSD repair (Higham) is applied if the
pairwise matrix is not PSD. PR = (Σλ)²/Σλ² of the correlation matrix.

VFP enters three ways (robustness): Option C (composite column via the hurdle's
law-of-total-covariance row, §5 — PRIMARY), Option B (presence rate r_i only), and
exclude-VFP. With φ=1 on TIMIT (every speaker creaks) Option C's presence term is
0, so the composite reduces to the direct moments of v_i = r_i·M_i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import (transformed_matrix, pr_from_cov, fisher_components)
from src.vfp_hurdle import (fit_hurdle, vfp_sigma_b_row, vfp_presence_only,
                            vfp_utt_composite)


# ── Σ_b assembly ────────────────────────────────────────────────────────────────

def _speaker_means(X: np.ndarray, sp: np.ndarray, speakers: np.ndarray) -> np.ndarray:
    """Vectorised NaN-aware per-speaker means (pandas groupby). (S, p)."""
    dfX = pd.DataFrame(X)
    dfX["_spk"] = sp
    g = dfX.groupby("_spk").mean()            # mean skips NaN per cell
    return g.reindex(speakers).to_numpy()


def pairwise_cov(M: np.ndarray) -> np.ndarray:
    """Pairwise-complete covariance (ddof=0). M (S,p) may contain NaN."""
    p = M.shape[1]
    S = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            m = np.isfinite(M[:, i]) & np.isfinite(M[:, j])
            if m.sum() > 1:
                a = M[m, i] - M[m, i].mean()
                b = M[m, j] - M[m, j].mean()
                c = float((a * b).mean())
            else:
                c = 0.0
            S[i, j] = S[j, i] = c
    return S


def nearest_psd(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.clip(w, 0.0, None)) @ V.T


def assemble(df, cfg, sp=None) -> dict:
    """Per-speaker quantities for the pooled (or relabelled) data."""
    if sp is None:
        sp = df[cfg["speaker_key"]].to_numpy()
    speakers = np.array(sorted(np.unique(sp)))
    X, names = transformed_matrix(df, cfg)
    Xbar = _speaker_means(X, sp, speakers)
    h = fit_hurdle(sp, df[cfg["vfp"]["name"]].to_numpy(float), speakers)
    return dict(speakers=speakers, names=names, X=X, Xbar=Xbar, h=h, sp=sp)


def sigma_b(A: dict, vfp: str = "C"):
    """Return (Sigma, names) for vfp in {'C' (Option C), 'B' (presence), 'none'}."""
    Xbar, h, names = A["Xbar"], A["h"], list(A["names"])
    if vfp == "none":
        return pairwise_cov(Xbar), names
    if vfp == "B":
        M = np.column_stack([vfp_presence_only(h), Xbar])
        return pairwise_cov(M), ["VFI(presence)"] + names
    # Option C: VFP row reconstructed, X block pairwise
    var_v, cov_vX = vfp_sigma_b_row(h, Xbar)
    SX = pairwise_cov(Xbar)
    p = Xbar.shape[1]
    S = np.zeros((p + 1, p + 1))
    S[0, 0] = var_v
    S[0, 1:] = cov_vX
    S[1:, 0] = cov_vX
    S[1:, 1:] = SX
    return S, ["VFI(optC)"] + names


# ── PR + diagnostics ────────────────────────────────────────────────────────────

def screen(S: np.ndarray, names: list, floor: float):
    sd = np.sqrt(np.clip(np.diag(S), 0.0, None))
    keep = sd >= floor
    dropped = [n for n, k in zip(names, keep) if not k]
    S2 = S[np.ix_(keep, keep)]
    repaired = False
    if np.linalg.eigvalsh((S2 + S2.T) / 2).min() < -1e-10:
        S2 = nearest_psd(S2)
        repaired = True
    return S2, dropped, repaired, int(keep.sum())


def corr(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
    return S / np.outer(d, d)


def crosscheck_dims(S: np.ndarray) -> dict:
    """90%-variance dim and spectral-entropy effective dim of the correlation matrix."""
    w = np.clip(np.linalg.eigvalsh(corr(S)), 0.0, None)[::-1]
    cum = np.cumsum(w) / w.sum()
    dim90 = int(np.searchsorted(cum, 0.90) + 1)
    pk = w / w.sum()
    pk = pk[pk > 0]
    H = -(pk * np.log(pk)).sum()
    return dict(dim_90pct_var=dim90, dim_spectral_entropy=float(np.exp(H)))


def pr_point(df, cfg, vfp="C"):
    A = assemble(df, cfg)
    S, names = sigma_b(A, vfp)
    S2, dropped, repaired, kept = screen(S, names, cfg["pr"]["sigma_b_floor"])
    return dict(PR=pr_from_cov(S2), kept=kept, dropped=dropped, repaired=repaired,
                crosscheck=crosscheck_dims(S2), A=A)


def pr_null(df, cfg, n_perm, seed, vfp="C"):
    rng = np.random.default_rng(seed)
    base_sp = df[cfg["speaker_key"]].to_numpy()
    out = []
    for _ in range(n_perm):
        A = assemble(df, cfg, sp=rng.permutation(base_sp))
        S, names = sigma_b(A, vfp)
        S2, *_ = screen(S, names, cfg["pr"]["sigma_b_floor"])
        out.append(pr_from_cov(S2))
    return np.array(out)


def pr_bootstrap(A, cfg, n_boot, seed, vfp="C"):
    """Bootstrap over speakers. With φ=1 Option C == direct moments of [v_i, Xbar]."""
    h, Xbar = A["h"], A["Xbar"]
    if vfp == "C":
        P = np.column_stack([h.v, Xbar]); names = ["VFI(optC)"] + list(A["names"])
    elif vfp == "B":
        P = np.column_stack([vfp_presence_only(h), Xbar]); names = ["VFI(presence)"] + list(A["names"])
    else:
        P = Xbar; names = list(A["names"])
    rng = np.random.default_rng(seed)
    S = P.shape[0]
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, S, S)
        Sig, *_ = screen(pairwise_cov(P[idx]), names, cfg["pr"]["sigma_b_floor"])
        out.append(pr_from_cov(Sig))
    return np.percentile(out, [2.5, 97.5]), float(np.mean(out))


def fisher_order(df, cfg, A):
    """Fisher F* per feature (VFP via its utterance composite); decreasing order."""
    sp, speakers = A["sp"], A["speakers"]
    rows = []
    for j, name in enumerate(A["names"]):
        fc = fisher_components(A["X"][:, j], sp, speakers)
        rows.append((name, fc["Fstar"]))
    z = vfp_utt_composite(df[cfg["vfp"]["name"]].to_numpy(float))
    rows.append(("VFI", fisher_components(z, sp, speakers)["Fstar"]))
    rows.sort(key=lambda t: (-(t[1] if np.isfinite(t[1]) else -1)))
    return rows
