"""Stage 2 of the Frauscher 2018 pipeline: cluster channels and identify the no-peak set."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise


Summary = Literal["mean", "median"]


def _summary(data, how: Summary, axis: int = 0) -> np.ndarray:
    """Mean or median along `axis`, returned as ndarray. Works on DataFrame or ndarray."""
    func = {"mean": np.mean, "median": np.median}.get(how)
    if func is None:
        raise ValueError(f"summary must be 'mean' or 'median', got {how!r}")
    return np.asarray(func(data, axis=axis))


def compute_clusters(
    psd_df: pd.DataFrame, n_clusters: int, random_seed: int = 2
) -> pd.DataFrame:
    """Cluster channels by their spectral profile using k-means.

    Returns the input DataFrame with a 'clusters' column appended.
    """
    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=n_clusters, max_iter=300, n_init=100, random_state=random_seed
    )
    kmeans.fit(psd_df.values)
    return psd_df.assign(clusters=kmeans.labels_)


def identify_small_clusters(
    psd_clust: pd.DataFrame, summary: Summary = "mean",
) -> list[int]:
    """Return cluster indices whose per-cluster summary spectrum is below all others' max at every frequency.

    Paper criterion (Frauscher 2018): the no-peak group's *mean* normalized
    spectrum is lower than the maximum among the other groups at every
    frequency. Pass summary='median' to use median instead (not paper-faithful).
    """
    cluster_summary = (
        psd_clust.drop(columns="clusters")
        .groupby(psd_clust["clusters"])
        .agg(summary)
    )

    smal = []
    n_freqs = cluster_summary.shape[1]
    for index, row in cluster_summary.iterrows():
        max_completion = cluster_summary.iloc[cluster_summary.index != index, :].max()
        if np.sum(np.less(row, max_completion)) == n_freqs:
            smal.append(index)
    return smal


def get_no_peak(
    psd_clust: pd.DataFrame, smal: list[int], summary: Summary = "mean",
) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """Take the 50% of channels in the no-peak cluster closest to its center.

    Paper uses the mean as center; pass summary='median' to use the median instead.
    Picks smal[0] when smal has multiple entries — to be revisited.
    Returns (None, None) when smal is empty (no cluster qualified as no-peak).
    """
    if not smal:
        return None, None
    no_peak = (
        psd_clust[psd_clust["clusters"] == smal[0]]
        .drop(columns="clusters")
        .copy()
    )
    freq_values = no_peak.values
    center = _summary(freq_values, summary, axis=0)
    temp = np.tile(center, (len(no_peak), 1))
    no_peak["distance_to_center"] = sklearn.metrics.pairwise.paired_distances(
        freq_values, temp
    )
    df = no_peak[
        no_peak["distance_to_center"] < no_peak["distance_to_center"].quantile(0.50)
    ]
    return df, center


def cluster_peak_frequencies(
    psd_clust: pd.DataFrame,
    f: np.ndarray,
    summary: Summary = "mean",
    baseline: np.ndarray | None = None,
    freq_range: tuple[float, float] | None = None,
) -> dict[int, float]:
    """Frequency of maximum (relative) power per cluster.

    Frauscher 2018 identifies a cluster's characteristic band as the
    frequencies where the cluster's spectrum exceeds the no-peak baseline.
    Pass ``baseline`` (e.g. no-peak cluster summary) to find the freq where
    each cluster stands out most above baseline — this picks up genuine
    spectral peaks rather than the 1/f maximum.

    Parameters
    ----------
    psd_clust : DataFrame
        Output of ``compute_clusters`` (must contain a ``clusters`` column).
    f : array
        Frequency grid matching the spectral columns.
    summary : 'mean' | 'median'
        Aggregation per cluster.
    baseline : array, shape (n_freqs,), optional
        Reference spectrum (e.g. ``get_no_peak`` center). When given,
        argmax is taken over ``cluster / baseline``. When None, raw
        argmax of cluster summary is used (likely returns a low-freq
        1/f point).
    freq_range : (fmin, fmax), optional
        Restrict argmax to this band. Useful to exclude the δ shoulder
        from being picked as "the" peak.

    Returns
    -------
    dict mapping cluster id → peak frequency in Hz.
    """
    cluster_summary = (
        psd_clust.drop(columns="clusters")
        .groupby(psd_clust["clusters"])
        .agg(summary)
    )
    f = np.asarray(f, dtype=float)
    arr = cluster_summary.to_numpy()
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=float)
        arr = arr / baseline[None, :]
    if freq_range is not None:
        fmin, fmax = freq_range
        mask = (f >= fmin) & (f <= fmax)
    else:
        mask = np.ones_like(f, dtype=bool)
    f_masked = f[mask]
    return {
        int(k): float(f_masked[arr[i, mask].argmax()])
        for i, k in enumerate(cluster_summary.index)
    }


def order_clusters_by_peak(
    psd_clust: pd.DataFrame,
    f: np.ndarray,
    summary: Summary = "mean",
    exclude: list[int] | None = None,
    baseline: np.ndarray | None = None,
    freq_range: tuple[float, float] | None = None,
) -> list[int]:
    """Cluster ids ordered by ascending peak frequency.

    Same ``baseline`` / ``freq_range`` semantics as
    ``cluster_peak_frequencies``. Pass ``exclude=[no_peak_id]`` to drop the
    no-peak cluster (it has no meaningful peak).
    """
    exclude = set(exclude or [])
    peaks = cluster_peak_frequencies(
        psd_clust, f, summary=summary,
        baseline=baseline, freq_range=freq_range,
    )
    return sorted([k for k in peaks if k not in exclude], key=lambda k: peaks[k])


def _elbow_scores_psd(psd_df: pd.DataFrame, n: int) -> dict[int, float]:
    from sklearn.cluster import KMeans

    psd_df = psd_df[0:160]
    sse = {}
    for k in range(1, n):
        kmeans = KMeans(n_clusters=k, max_iter=100).fit(psd_df)
        sse[k] = kmeans.inertia_
    return sse


def eblow_psd(psd_df: pd.DataFrame, n: int) -> None:
    import matplotlib.pyplot as plt

    sse = _elbow_scores_psd(psd_df, n)
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()