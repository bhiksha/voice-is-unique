"""timit_features — reproducible 40-feature utterance-level extractor for TIMIT.

Nothing here computes feature math or touches the corpus at scale until the
CONFIG block (config.py) and the open methodology questions (DECISIONS.md) have
been reviewed and approved. See the verification protocol in the project brief.
"""
from timit_features.config import CONFIG, FEATURE_NAMES

__all__ = ["CONFIG", "FEATURE_NAMES"]
