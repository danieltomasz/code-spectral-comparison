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