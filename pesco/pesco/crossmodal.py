"""Subject-clustered bootstrap for cross-modal per-cell spectral comparisons.

The unit of analysis is a *cell* (e.g. a lobe × frequency-bin mean across
channels). Channels are nested in subjects, so every bootstrap resamples whole
**subjects** (with replacement), never channels — channel-level resampling
understates uncertainty when a few subjects contribute many channels.

All functions are pure NumPy and NaN-aware (use NaN to mark "channel does not
contribute to this cell", e.g. a different lobe).
"""

from __future__ import annotations

import numpy as np


def bin_spectrum(
    freqs: np.ndarray,
    values: np.ndarray,
    bin_width: float,
    fmin: float | None = None,
    fmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average a ``channels × freq`` matrix into fixed-width frequency bins.

    Parameters
    ----------
    freqs : (F,) array
        Frequency grid of the columns of ``values``.
    values : (N, F) array
        Per-channel spectra.
    bin_width : float
        Bin width in Hz (e.g. 2.0).
    fmin, fmax : float, optional
        Binning range; default to ``freqs`` min/max.

    Returns
    -------
    centers : (B,) array
        Bin-centre frequencies.
    binned : (N, B) array
        Mean of ``values`` over the frequencies in each bin (NaN-aware).
    """
    freqs = np.asarray(freqs, dtype=float)
    values = np.asarray(values, dtype=float)
    fmin = float(freqs.min()) if fmin is None else fmin
    fmax = float(freqs.max()) if fmax is None else fmax
    edges = np.arange(fmin, fmax + bin_width, bin_width)
    idx = np.digitize(freqs, edges) - 1
    n_bins = len(edges) - 1
    binned = np.full((values.shape[0], n_bins), np.nan)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            binned[:, b] = np.nanmean(values[:, mask], axis=1)
    centers = edges[:-1] + bin_width / 2
    return centers, binned


def _subject_index(subjects: np.ndarray) -> dict:
    subjects = np.asarray(subjects)
    return {u: np.where(subjects == u)[0] for u in np.unique(subjects)}


def cluster_bootstrap_cell_means(
    matrix: np.ndarray,
    subjects: np.ndarray,
    B: int = 2000,
    ci: float = 95,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell channel mean with a subject-clustered percentile CI.

    Parameters
    ----------
    matrix : (N, C) array
        Channel × cell values (NaN where a channel does not enter a cell).
    subjects : (N,) array
        Subject id per channel (the cluster unit).
    B, ci, seed : bootstrap controls.

    Returns
    -------
    point, lo, hi : (C,) arrays
        Channel mean per cell and its ``ci`` percentile interval over
        subject-resampled means.
    """
    matrix = np.asarray(matrix, dtype=float)
    point = np.nanmean(matrix, axis=0)
    groups = _subject_index(subjects)
    uniq = np.array(list(groups))
    rng = np.random.default_rng(seed)
    boots = np.empty((B, matrix.shape[1]))
    for b in range(B):
        rows = np.concatenate([groups[u] for u in rng.choice(uniq, uniq.size, True)])
        boots[b] = np.nanmean(matrix[rows], axis=0)
    lo, hi = np.nanpercentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2], axis=0)
    return point, lo, hi


def subject_means(matrix: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a channel × cell matrix to subject × cell means (NaN-aware).

    The subject is the exchangeable unit for the cluster-permutation test, so
    each subject contributes one row (the mean over its channels).
    """
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)
    rows = [np.nanmean(matrix[subjects == u], axis=0) for u in uniq]
    return uniq, np.vstack(rows)


def _welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column Welch t between two subject × cell matrices."""
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    denom = np.sqrt(va / a.shape[0] + vb / b.shape[0])
    return np.where(denom > 0, (ma - mb) / denom, 0.0)


def _contiguous_clusters(stat: np.ndarray, threshold: float):
    """Runs of adjacent cells with |stat| > threshold and constant sign."""
    above = np.abs(stat) > threshold
    clusters, i, n = [], 0, len(stat)
    while i < n:
        if above[i]:
            j, sign = i, np.sign(stat[i])
            while j + 1 < n and above[j + 1] and np.sign(stat[j + 1]) == sign:
                j += 1
            idx = np.arange(i, j + 1)
            clusters.append((idx, float(stat[idx].sum())))
            i = j + 1
        else:
            i += 1
    return clusters


def cluster_permutation_freq(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_perm: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Maris–Oostenveld cluster-permutation test across frequency, subject-level.

    Two independent cohorts (rows = subjects, columns = frequency bins). A Welch
    t per bin is thresholded at the two-sided ``alpha`` critical value to form
    contiguous same-sign clusters; cluster mass is the summed t. The null is the
    max |cluster mass| over ``n_perm`` random relabellings of the pooled subjects
    into the two group sizes — which controls the family-wise error across the
    frequency axis (unlike per-bin CIs).

    Returns ``{t_obs, threshold, sig (bool per bin), clusters [(idx, mass, p)]}``.
    Subjects are the exchangeable unit, so this honours the channels-in-subjects
    nesting. Requires >= 2 subjects per group.
    """
    from scipy.stats import t as _tdist

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    na, nb, n_bins = a.shape[0], b.shape[0], a.shape[1]
    if na < 2 or nb < 2:
        raise ValueError("Need >= 2 subjects per group for the permutation test.")

    threshold = float(_tdist.ppf(1 - alpha / 2, na + nb - 2))
    t_obs = _welch_t(a, b)
    observed = _contiguous_clusters(t_obs, threshold)

    pooled = np.vstack([a, b])
    n = na + nb
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_perm)
    for p in range(n_perm):
        idx = rng.permutation(n)
        tp = _welch_t(pooled[idx[:na]], pooled[idx[na:]])
        cl = _contiguous_clusters(tp, threshold)
        null_max[p] = max((abs(m) for _, m in cl), default=0.0)

    sig = np.zeros(n_bins, dtype=bool)
    clusters = []
    for idx, mass in observed:
        pval = (np.sum(null_max >= abs(mass)) + 1) / (n_perm + 1)
        clusters.append((idx, mass, float(pval)))
        if pval < alpha:
            sig[idx] = True
    return {"t_obs": t_obs, "threshold": threshold, "sig": sig, "clusters": clusters}


def two_sample_cluster_diff(
    matrix_a: np.ndarray,
    subjects_a: np.ndarray,
    matrix_b: np.ndarray,
    subjects_b: np.ndarray,
    B: int = 2000,
    ci: float = 95,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Difference of per-cell means (A − B) with a two-sample clustered CI.

    The two modalities are independent cohorts with different subjects, so each
    is resampled at the subject level **independently** and the per-cell means
    differenced within each replicate. Cells must be aligned (same columns).

    Returns
    -------
    point, lo, hi : (C,) arrays
        ``mean_A − mean_B`` per cell and its CI.
    significant : (C,) bool array
        True where the CI excludes 0.
    """
    matrix_a = np.asarray(matrix_a, dtype=float)
    matrix_b = np.asarray(matrix_b, dtype=float)
    point = np.nanmean(matrix_a, axis=0) - np.nanmean(matrix_b, axis=0)
    ga, gb = _subject_index(subjects_a), _subject_index(subjects_b)
    ua, ub = np.array(list(ga)), np.array(list(gb))
    rng = np.random.default_rng(seed)
    boots = np.empty((B, matrix_a.shape[1]))
    for b in range(B):
        ra = np.concatenate([ga[u] for u in rng.choice(ua, ua.size, True)])
        rb = np.concatenate([gb[u] for u in rng.choice(ub, ub.size, True)])
        boots[b] = np.nanmean(matrix_a[ra], axis=0) - np.nanmean(matrix_b[rb], axis=0)
    lo, hi = np.nanpercentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2], axis=0)
    significant = (lo > 0) | (hi < 0)
    return point, lo, hi, significant
