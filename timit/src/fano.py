"""Fano — joint-information LOWER bound, all 6300 utterances (§4).

A speaker classifier is trained with utterance-disjoint, speaker-stratified 5-fold
CV (every speaker in every fold; no utterance in both train and test). No row is
ever listwise-deleted and missingness is never a cue:
  - genuine state-3 cells → imputed to the TRAIN-fold mean, with NO indicator column
    (mean imputation injects no speaker info; omitting the indicator blocks any
    "was-missing" cue). All 6300 rows are kept ⇒ ceiling H(Y)=log2 S.
  - VFP enters as two genuine inputs: the per-speaker presence rate r_i (estimated
    on TRAIN folds only) and the z-scored log-magnitude (state-1); state-2 zeros'
    magnitude is mean-imputed and the presence rate carries the zero signal — no
    state-3 indicator (VFP has no state-3 in TIMIT anyway).
Bounds per fold/classifier:
  P_e = 1-acc;  I_fano = H(Y) - H_b(P_e) - P_e·log2(S-1)
  I_xent = H(Y) - mean_test(-log2 q_true),  q clipped to >=1e-12
Headline = max over classifiers for each bound. Imputation is fit on TRAIN only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from src.common import state_codes, transform_of


def make_static_design(df, cfg):
    """Fold-independent design: transformed non-VFP features + VFP log-magnitude.
    NaN is preserved (state-3 cells, and state-2 for VFP_logmag) for per-fold impute."""
    vfp = cfg["vfp"]["name"]
    cols, names = [], []
    for f in cfg["feature_names"]:
        if f == vfp:
            continue
        v = df[f].to_numpy(float)
        if transform_of(f, cfg) == "log":
            v = np.log(v)
        cols.append(v)
        names.append(f)
    val = df[vfp].to_numpy(float)
    st = state_codes(val)
    lm = np.full(val.shape, np.nan)
    lm[st == 1] = np.log(val[st == 1])             # state-2/3 → NaN → train-mean imputed
    cols.append(lm)
    names.append("VFP_logmag")
    return np.column_stack(cols), names, val, df[cfg["speaker_key"]].to_numpy()


def presence_column(vfp_raw, sp, train_mask):
    """Per-speaker presence rate r_i estimated from TRAIN utterances; broadcast to all."""
    st = state_codes(vfp_raw)
    t = pd.DataFrame({"sp": sp,
                      "meas": ((st != 3) & train_mask).astype(float),
                      "s1": ((st == 1) & train_mask).astype(float)})
    g = t.groupby("sp")[["meas", "s1"]].sum()
    rate = (g["s1"] / g["meas"].replace(0, np.nan)).fillna(0.0)
    return pd.Series(sp).map(rate).to_numpy()


def state3_impute_counts(df, cfg):
    """Per-feature count of genuine state-3 (NaN) cells that get mean-imputed."""
    out = {}
    for f in cfg["feature_names"]:
        v = df[f].to_numpy(float)
        out[f] = int(np.isnan(v).sum())
    return out


def _impute_scale(Dtr, Dte):
    mu = np.nanmean(Dtr, axis=0)
    Dtr = np.where(np.isnan(Dtr), mu, Dtr)
    Dte = np.where(np.isnan(Dte), mu, Dte)
    sc = StandardScaler().fit(Dtr)
    return sc.transform(Dtr), sc.transform(Dte)


def _classifier(name, S):
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
    if name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=120,
                             random_state=0)
    raise ValueError(name)


def fano_bounds(y_true, y_pred, proba, classes, S):
    HY = np.log2(S)
    acc = float((y_true == y_pred).mean())
    Pe = 1.0 - acc
    Hb = 0.0 if Pe in (0.0, 1.0) else -(Pe * np.log2(Pe) + (1 - Pe) * np.log2(1 - Pe))
    I_fano = HY - Hb - Pe * np.log2(S - 1)
    cls = {c: i for i, c in enumerate(classes)}
    q = np.clip(np.array([proba[i, cls[t]] for i, t in enumerate(y_true)]), 1e-12, 1.0)
    I_xent = HY - float(np.mean(-np.log2(q)))
    return dict(acc=acc, I_fano=float(I_fano), I_xent=float(I_xent))


def run_fold(D, names, vfp_raw, sp, y, train_idx, test_idx, S, clf_name, shuffle_y=None):
    train_mask = np.zeros(len(y), bool); train_mask[train_idx] = True
    pres = presence_column(vfp_raw, sp, train_mask)
    Dfull = np.column_stack([D, pres])                 # append VFP presence rate
    Dtr, Dte = _impute_scale(Dfull[train_idx], Dfull[test_idx])
    yy = y if shuffle_y is None else shuffle_y
    clf = _classifier(clf_name, S).fit(Dtr, yy[train_idx])
    proba = clf.predict_proba(Dte)
    pred = clf.classes_[proba.argmax(1)]
    return fano_bounds(yy[test_idx], pred, proba, clf.classes_, S)


def cv_splits(y, n_folds, seed):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
