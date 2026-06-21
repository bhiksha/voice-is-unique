"""CONFIG for the Fisher-ratio / participation-ratio analysis stage.

Every choice that moves a number is here (prompt §6). Values flagged PROPOSED are
defaults awaiting confirmation at verification Step 1.

NOTE: the prompt names the creak feature "VFP"; in our extracted data it is
"VFI" (renamed earlier). The mapping is VFP == VFI. VFI is the proportion of
voiced frames that are creaky: 0 (no fry) is a valid measurement, not "missing",
so it is analysed RAW (transform "none"). NaN occurs only for utterances with no
voiced frames. (An earlier `log_nonzero` transform mapped 0 -> NaN, which wrongly
discarded the ~489 zero-fry utterances; removed 2026-06-18.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from timit_features.config import FEATURE_NAMES


def _default_transforms() -> dict[str, str]:
    t = {f: "none" for f in FEATURE_NAMES}
    # VFI (prompt's "VFP") stays RAW: 0 = no fry is a valid value, not missing.
    t["NAQ"] = "log"                  # natural log of raw value
    # diagnose set resolved to log after §a.1 (all positive, no zeros, skew 5-7
    # -> ~0 under log); approved 2026-06-16.
    for f in ("alpha_ratio", "LHR", "SPI"):
        t[f] = "log"
    return t


@dataclass(frozen=True)
class AnalysisConfig:
    input_parquet: str = "~/data/timit-feats/all_utterances.parquet"
    speaker_key: str = "speaker_id"

    # (a) per-feature transform, then corpus-wide z-score to unit total variance
    transform_per_feature: dict = field(default_factory=_default_transforms)
    log_eps: float = 0.0              # log features: pure log (no eps); non-positives -> ASK

    # (c) degenerate-feature screen + PR
    sigma_b_floor: float = 1e-3       # PROPOSED: drop features with between-speaker SD (in
                                      #   z-scored units) below this before the correlation PR
    missing_data_rule: str = "pairwise"   # primary; listwise also reported
    psd_repair: str = "nearest_psd_higham"
    between_variance_estimator: str = "variance_components"  # primary; var-of-means reported

    # bootstrap (resample unit = speakers)
    n_boot: int = 1000                # PROPOSED
    bootstrap_seed: int = 0

    # saturation / inclusion / tolerances
    saturation_delta: float = 0.05    # PROPOSED: ΔPR below this ⇒ "non-redundant feature" count
    min_utts_per_speaker: int = 1     # PROPOSED: include all; within-var pooling handles n_i<2
    invariance_tol: float = 1e-9      # F*/PR must match pre/post z-score to this
    pr_crosscheck_tol: float = 0.5    # PR vs 90%-cutoff / spectral-entropy agreement tolerance

    out_dir: str = "~/data/timit-feats/analysis"


ANALYSIS_CONFIG = AnalysisConfig()
