"""Downstream analysis stage: per-feature transforms, Fisher ratios (F* =
variance-components ratio), and incremental participation ratio (PR).

Consumes timit-feats/all_utterances.parquet; never recomputes audio features.
See CLAUDE_PROMPT_fisher_pr_report_v2 and DECISIONS for the methodology.
"""
