from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

FloatArray = NDArray[np.float64]


def regional_permtest(
    x: FloatArray,
    y: FloatArray,
    n_perm: int = 10_000,
    seed: int | None = None,
) -> dict:
    """Permutation-based Spearman correlation for cross-modal regional comparisons.

    Shuffles region labels ``n_perm`` times and recomputes Spearman ρ each time
    to build a null distribution.  The empirical p-value is the proportion of
    permuted ρ values that are greater than or equal to the observed ρ (one-tailed,
    testing for positive correspondence).

    Parameters
    ----------
    x, y : array-like of shape (n_regions,)
        Regional parameter vectors to compare (e.g. exponent across 38 regions).
        NaN-containing rows are dropped pairwise before computing ρ.
    n_perm : int, optional, default: 10_000
        Number of permutations.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``rho``       — observed Spearman ρ
        ``p_analytic`` — two-tailed analytic p-value from scipy
        ``p_perm``    — one-tailed permutation p-value
        ``n``         — number of non-NaN region pairs used
        ``null``      — array of shape (n_perm,) containing null ρ values

    Examples
    --------
    >>> result = regional_permtest(ieeg_exponents, source_exponents, seed=42)
    >>> print(f"rho={result['rho']:.3f}, p_perm={result['p_perm']:.4f}")
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x_v = x[valid]
    y_v = y[valid]
    n = int(valid.sum())

    if n < 3:
        raise ValueError(
            f"Need at least 3 non-NaN region pairs; got {n}. "
            "Check for NaN values in x or y."
        )

    rho, p_analytic = spearmanr(x_v, y_v)

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i], _ = spearmanr(x_v, y_v[perm])

    p_perm = float(np.mean(null >= rho))

    return {
        "rho": float(rho),
        "p_analytic": float(p_analytic),
        "p_perm": p_perm,
        "n": n,
        "null": null,
    }
