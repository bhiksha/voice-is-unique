"""VFP hurdle (Option C) — presence/magnitude split and the law-of-total-covariance
reconstruction of VFP's between-speaker covariance row.

Per speaker i (over that speaker's MEASURABLE utterances, states 1∪2; state-3
excluded from numerator and denominator):
    presence rate   r_i = (# state-1 utts) / (# measurable utts)
    magnitude       M_i = mean of z-scored log(VFP) over speaker i's state-1 utts
                          (undefined when r_i = 0 — a non-creaker has no magnitude)

The composite per-speaker VFP coordinate that occupies one row of Σ_b is
    v_i = r_i · M_i          (and v_i = 0 when r_i = 0, i.e. non-creakers enter via
                              presence only; no magnitude is ever imputed for them)

This is exactly the speaker mean of the per-utterance composite Z = 1{present}·z-log(VFP)
(Z = 0 on absent utts), so Var_b/Cov_b of v_i ARE the between-speaker moments of VFP.

Option C writes those moments through the law of total covariance with the
creaker/non-creaker split P = 1{r_i>0}.  With E[v|absent]=0, Var(v|absent)=0:

    Var_b(VFP)      = φ·Var(v|present)            +  φ(1-φ)·E[v|present]²
                      └ prevalence-weighted          └ presence/absence split
                        within-creaker variance
    Cov_b(VFP, X_j) = φ·Cov(v, X_j|present)       +  φ(1-φ)·E[v|present]·(E[X_j|present]-E[X_j|absent])

φ = P(present).  These are the law-of-total-(co)variance identities; the within-
present term already carries the presence×magnitude coupling ("cross terms").
``decompose_var``/``decompose_cov`` evaluate the right-hand side from the hurdle
pieces; the synthetic-recovery test (tests/test_vfp_hurdle.py) confirms they match
the direct moments and that PR with the composite column matches PR on
fully-observed data from the same generative model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common import state_codes, pr_from_cov


@dataclass
class Hurdle:
    speakers: np.ndarray   # (S,) speaker ids
    r: np.ndarray          # (S,) presence rate
    M: np.ndarray          # (S,) mean z-log-magnitude over state-1 (nan if r=0)
    v: np.ndarray          # (S,) composite v_i = r_i·M_i (0 when r_i=0)
    present: np.ndarray    # (S,) bool r_i>0
    n_state1: np.ndarray   # (S,) #creaky utts
    n_meas: np.ndarray     # (S,) #measurable utts (states 1∪2)


def fit_hurdle(speaker_ids: np.ndarray, vfp_values: np.ndarray,
               speakers: np.ndarray | None = None) -> Hurdle:
    """Fit the hurdle from per-utterance speaker ids and raw VFP values.

    Magnitude = natural log of state-1 values, z-scored corpus-wide over state-1.
    """
    speaker_ids = np.asarray(speaker_ids)
    val = np.asarray(vfp_values, dtype=float)
    st = state_codes(val)
    s1 = st == 1

    mag = np.full(val.shape, np.nan)
    mag[s1] = np.log(val[s1])
    mu, sd = mag[s1].mean(), mag[s1].std()
    zmag = (mag - mu) / (sd if sd > 0 else 1.0)

    if speakers is None:
        speakers = np.array(sorted(np.unique(speaker_ids)))
    S = len(speakers)
    r = np.zeros(S)
    M = np.full(S, np.nan)
    n1 = np.zeros(S, dtype=int)
    nm = np.zeros(S, dtype=int)
    for k, s in enumerate(speakers):
        idx = speaker_ids == s
        meas = idx & (st != 3)
        s1i = idx & s1
        nm[k] = int(meas.sum())
        n1[k] = int(s1i.sum())
        if nm[k] > 0:
            r[k] = n1[k] / nm[k]
        if n1[k] > 0:
            M[k] = zmag[s1i].mean()
    v = np.where(r > 0, r * np.nan_to_num(M, nan=0.0), 0.0)
    return Hurdle(speakers, r, M, v, r > 0, n1, nm)


# ── Option C: law-of-total-covariance reconstruction of VFP's Σ_b row ───────────

def decompose_var(h: Hurdle) -> tuple[float, dict]:
    """Var_b(VFP) via the presence split. Returns (value, term breakdown)."""
    P = h.present
    phi = float(P.mean())
    vp = h.v[P]
    Evp = float(vp.mean())
    Varvp = float(vp.var())                      # ddof=0
    term_magnitude = phi * Varvp
    term_presence = phi * (1.0 - phi) * Evp ** 2
    return term_magnitude + term_presence, dict(
        phi=phi, E_v_present=Evp, Var_v_present=Varvp,
        term_magnitude=term_magnitude, term_presence=term_presence)


def decompose_cov(h: Hurdle, x: np.ndarray) -> float:
    """Cov_b(VFP, X_j) via the presence split, pairwise over speakers measurable on x."""
    P = h.present & np.isfinite(x)
    A = (~h.present) & np.isfinite(x)            # absent speakers measurable on x
    phi = float(P.mean()) if (P.sum() + A.sum()) else 0.0
    if P.sum() == 0:
        return 0.0
    vp, xp = h.v[P], x[P]
    Evp = float(vp.mean())
    cov_present = float(((vp - Evp) * (xp - xp.mean())).mean())
    xa_mean = float(x[A].mean()) if A.any() else 0.0
    term1 = phi * cov_present
    term2 = phi * (1.0 - phi) * Evp * (float(xp.mean()) - xa_mean)
    return term1 + term2


def vfp_sigma_b_row(h: Hurdle, Xbar: np.ndarray):
    """Return (var_vfp, cov_vfp_X) for the composite VFP row of Σ_b via Option C."""
    var_vfp, _ = decompose_var(h)
    cov = np.array([decompose_cov(h, Xbar[:, j]) for j in range(Xbar.shape[1])])
    return var_vfp, cov


# ── PR robustness representations of VFP ────────────────────────────────────────

def vfp_presence_only(h: Hurdle) -> np.ndarray:
    """Option B: VFP represented by per-speaker presence rate r_i only."""
    return h.r.copy()


def vfp_utt_composite(vfp_values: np.ndarray) -> np.ndarray:
    """Per-utterance VFP composite Z used for Fisher ordering and bootstrap:
    state-1 → z-scored log(VFP); state-2 (zero) → 0; state-3 → NaN. The speaker
    mean of Z equals the composite v_i = r_i·M_i."""
    val = np.asarray(vfp_values, dtype=float)
    st = state_codes(val)
    z = np.full(val.shape, np.nan)
    s1 = st == 1
    lm = np.log(val[s1])
    z[s1] = (lm - lm.mean()) / (lm.std() if lm.std() > 0 else 1.0)
    z[st == 2] = 0.0
    return z
