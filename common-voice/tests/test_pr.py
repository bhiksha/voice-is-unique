"""PR sanity: duplicate column → PR=1; independent columns → PR≈d."""
import numpy as np

from src.common import pr_from_cov


def test_duplicate_column_pr_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 1))
    M = np.hstack([x, x, x])                 # three identical columns
    Sig = np.cov(M, rowvar=False)
    assert abs(pr_from_cov(Sig) - 1.0) < 1e-6


def test_independent_columns_pr_near_d():
    rng = np.random.default_rng(0)
    d = 8
    M = rng.standard_normal((20000, d))      # independent → R ≈ I → PR ≈ d
    assert abs(pr_from_cov(np.cov(M, rowvar=False)) - d) < 0.5
