"""§a.1 zero-inflation / scale diagnostic for all 40 features (on RAW values)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import skew

from timit_features.config import FEATURE_NAMES

_SPIKE_FRAC = 0.02      # a single value holding >2% of the mass = a "spike"
_ZERO_FRAC = 0.02       # >2% exact zeros = zero-inflated candidate


def _flags(v: np.ndarray) -> dict:
    n = v.size
    if n == 0:
        return dict(spike_at_single_value=False, zero_inflated=False,
                    spreads_under_log=False, negative_values_present=False)
    vals, cnts = np.unique(v, return_counts=True)
    mode_frac = cnts.max() / n
    zero_frac = (v == 0).sum() / n
    nz = v[v > 0]
    raw_skew = float(skew(v)) if n > 2 else np.nan
    log_skew = float(skew(np.log(nz))) if nz.size > 2 else np.nan
    spreads = (np.isfinite(raw_skew) and np.isfinite(log_skew)
               and v.min() >= 0 and abs(raw_skew) > 2 and abs(log_skew) < abs(raw_skew))
    return dict(
        spike_at_single_value=bool(mode_frac > _SPIKE_FRAC and vals.size > 5),
        zero_inflated=bool(zero_frac > _ZERO_FRAC and nz.size > 20),
        spreads_under_log=bool(spreads),
        negative_values_present=bool(v.min() < 0),
    )


def diagnostic_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FEATURE_NAMES:
        v = df[f].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        n = v.size
        row = dict(
            feature=f, n=n,
            n_zero=int((v == 0).sum()) if n else 0,
            n_one=int((v == 1).sum()) if n else 0,
            min=float(v.min()) if n else np.nan,
            max=float(v.max()) if n else np.nan,
            frac_unique=(float(np.unique(v).size / n) if n else np.nan),
            skew=float(skew(v)) if n > 2 else np.nan,
        )
        row.update(_flags(v))
        rows.append(row)
    return pd.DataFrame(rows)


def save_histograms(df: pd.DataFrame, feats, outdir: str) -> list[str]:
    """Save linear + log-axis histograms (side by side) per feature; return paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for f in feats:
        v = df[f].to_numpy(dtype=float); v = v[np.isfinite(v)]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(v, bins=60, color="steelblue", edgecolor="white")
        ax[0].set_title(f"{f}  (linear, n={v.size})"); ax[0].set_xlabel(f)
        nz = v[v > 0]
        if nz.size:
            ax[1].hist(np.log(nz), bins=60, color="indianred", edgecolor="white")
        ax[1].set_title(f"{f}  (log of >0, n_pos={nz.size})"); ax[1].set_xlabel(f"log({f})")
        fig.tight_layout()
        p = os.path.join(outdir, f"hist_{f}.png"); fig.savefig(p, dpi=90); plt.close(fig)
        paths.append(p)
    return paths
