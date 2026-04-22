"""Stage 2 of the Frauscher 2018 pipeline: cluster channels and identify the no-peak set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise


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


def identify_small_clusters(cluster_medians: pd.DataFrame) -> list[int]:
    """Return cluster indices whose median spectrum is below all others' max at every frequency.

    Paper criterion for the no-peak group: its mean normalized spectrum must be
    lower than the maximum among the other groups at every frequency.
    (This implementation uses median, not mean — to be revisited.)
    """
    smal = []
    n_freqs = cluster_medians.shape[1]
    for index, row in cluster_medians.iterrows():
        max_completion = cluster_medians.iloc[cluster_medians.index != index, :].max()
        if np.sum(np.less(row, max_completion)) == n_freqs:
            smal.append(index)
    return smal


def get_no_peak(
    psd_clust: pd.DataFrame, smal: list[int]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Take the 50% of channels in the no-peak cluster closest to its median.

    Paper takes the 50% closest to the mean — this uses median. To be revisited.
    Picks smal[0] when smal has multiple entries — to be revisited.
    """
    no_peak = psd_clust[psd_clust["clusters"] == smal[0]]
    no_peak = no_peak[no_peak.columns[:-1]]
    median = np.median(no_peak, 0)
    temp_med = np.tile(median, (len(no_peak), 1))
    no_peak["distances_to_median"] = sklearn.metrics.pairwise.paired_distances(
        no_peak, temp_med
    )
    df = no_peak[
        no_peak.distances_to_median < no_peak.distances_to_median.quantile(0.50)
    ]
    return df, median


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
