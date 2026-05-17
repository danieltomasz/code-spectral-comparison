"""Stage 2 of the Frauscher 2018 pipeline: cluster channels and identify the no-peak set."""

from __future__ import annotations

from typing import Hashable, Iterable, Literal, NamedTuple

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise


Summary = Literal["mean", "median"]


class Band(NamedTuple):
    """Frequency band: half-open interval ``[lo, hi)`` plus display info."""
    lo: float
    hi: float
    name: str          # legend / dict label, e.g. "δ"
    tex: str           # mathtext form for axis overlay, e.g. r"\delta"
    label_x: float     # x-position for Greek letter overlay (log-axis)


# Standard EEG bands. ``label_x`` matches the historical positions used in
# ``plot_clusters`` so visual layout is preserved. Single source of truth
# for both x-tick edges and band assignment.
EEG_BANDS: tuple[Band, ...] = (
    Band(0.5, 4.0, "δ", r"\delta", 2.0),
    Band(4.0, 8.0, "θ", r"\theta", 6.0),
    Band(8.0, 13.0, "α", r"\alpha", 10.0),
    Band(13.0, 30.0, "β", r"\beta", 16.0),
    Band(30.0, 80.0, "γ", r"\gamma", 36.0),
)


def band_edges(bands: Iterable[Band] = EEG_BANDS) -> list[float]:
    """xtick positions: each band's lower edge plus final upper edge."""
    bands = list(bands)
    return [b.lo for b in bands] + [bands[-1].hi]


def cluster_bands(
    peaks: dict[int, float | None],
    bands: Iterable[Band] = EEG_BANDS,
    no_peak_label: str = "No peak",
) -> dict[int, str]:
    """Map cluster id → band name based on peak frequency.

    Input ``peaks`` is the output of ``cluster_peak_frequencies`` (with
    ``no_peak=smal`` so no-peak clusters map to None).
    """
    edges = list(bands)
    out: dict[int, str] = {}
    for k, v in peaks.items():
        if v is None:
            out[k] = no_peak_label
            continue
        for b in edges:
            if b.lo <= v < b.hi:
                out[k] = b.name
                break
        else:
            out[k] = f"{v:.1f} Hz"
    return out


def _resolve_feature_cols(
    df: pd.DataFrame, feature_cols: Iterable[Hashable] | None,
) -> list[Hashable]:
    """Return explicit feature columns, or auto-pick numeric-named ones."""
    if feature_cols is not None:
        return list(feature_cols)
    return [c for c in df.columns if isinstance(c, (int, float, np.floating))]


def _summary(data, how: Summary, axis: int = 0) -> np.ndarray:
    """Mean or median along `axis`, returned as ndarray. Works on DataFrame or ndarray."""
    func = {"mean": np.mean, "median": np.median}.get(how)
    if func is None:
        raise ValueError(f"summary must be 'mean' or 'median', got {how!r}")
    return np.asarray(func(data, axis=axis))


def compute_clusters(
    df: pd.DataFrame,
    n_clusters: int,
    random_seed: int = 2,
    feature_cols: Iterable[Hashable] | None = None,
    label_col: str = "clusters",
) -> pd.DataFrame:
    """Cluster rows by spectral profile. Preserves all non-feature columns."""
    from sklearn.cluster import KMeans

    cols = _resolve_feature_cols(df, feature_cols)
    X = df[cols].to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, max_iter=300, n_init=100, random_state=random_seed)
    return df.assign(**{label_col: kmeans.fit_predict(X)})


def identify_small_clusters(
    psd_clust: pd.DataFrame,
    feature_cols: Iterable[Hashable] | None = None,
    summary: Summary = "mean",
    label_col: str = "clusters",
) -> list[int]:
    """Return cluster indices whose per-cluster summary spectrum is below all others' max at every frequency.

    Paper criterion (Frauscher 2018): the no-peak group's *mean* normalized
    spectrum is lower than the maximum among the other groups at every
    frequency. Pass summary='median' to use median instead (not paper-faithful).
    """
    cols = _resolve_feature_cols(psd_clust.drop(columns=label_col), feature_cols)
    cluster_summary = (
        psd_clust[cols]
        .groupby(psd_clust[label_col])
        .agg(summary)
    )

    smal = []
    n_freqs = cluster_summary.shape[1]
    for index, row in cluster_summary.iterrows():
        max_completion = cluster_summary.iloc[cluster_summary.index != index, :].max()
        if np.sum(np.less(row, max_completion)) == n_freqs:
            smal.append(index)
    return smal


def find_k_with_no_peak(
    df: pd.DataFrame,
    k_range: range = range(2, 16),
    feature_cols: Iterable[Hashable] | None = None,
    summary: Summary = "mean",
    random_seed: int = 2,
    label_col: str = "clusters",
) -> tuple[int, pd.DataFrame, list[int]]:
    """Frauscher 2018 procedure: smallest k that yields a no-peak cluster.

    "By repeating the classification with increasing number of groups k,
    eventually a group without peaks was found." Returns the first k for
    which ``identify_small_clusters`` returns a non-empty list.

    Returns
    -------
    k : int
        The selected number of clusters.
    psd_clust : DataFrame
        Output of ``compute_clusters`` at that k.
    smal : list[int]
        No-peak cluster ids at that k.

    Raises
    ------
    ValueError if no k in ``k_range`` qualifies.
    """
    for k in k_range:
        psd_clust = compute_clusters(
            df, n_clusters=k, random_seed=random_seed,
            feature_cols=feature_cols, label_col=label_col,
        )
        smal = identify_small_clusters(
            psd_clust, feature_cols=feature_cols,
            summary=summary, label_col=label_col,
        )
        if smal:
            return k, psd_clust, smal
    raise ValueError(
        f"No k in {list(k_range)} produced a no-peak cluster (criterion={summary!r})."
    )


def get_no_peak(
    psd_clust: pd.DataFrame,
    smal: list[int],
    feature_cols: Iterable[Hashable] | None = None,
    summary: Summary = "mean",
    label_col: str = "clusters",
) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """Take the 50% of channels in the no-peak cluster closest to its center.

    Paper uses the mean as center; pass summary='median' to use the median instead.
    Picks smal[0] when smal has multiple entries — to be revisited.
    Returns (None, None) when smal is empty (no cluster qualified as no-peak).
    """
    if not smal:
        return None, None
    cols = _resolve_feature_cols(psd_clust.drop(columns=label_col), feature_cols)
    no_peak = psd_clust.loc[psd_clust[label_col] == smal[0], cols].copy()
    freq_values = no_peak.to_numpy()
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
    feature_cols: Iterable[Hashable] | None = None,
    label_col: str = "clusters",
    no_peak: Iterable[int] | None = None,
) -> dict[int, float | None]:
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
    cols = _resolve_feature_cols(psd_clust.drop(columns=label_col), feature_cols)
    cluster_summary = (
        psd_clust[cols]
        .groupby(psd_clust[label_col])
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
    no_peak_set = {int(k) for k in (no_peak or [])}
    return {
        int(k): (
            None if int(k) in no_peak_set
            else float(f_masked[arr[i, mask].argmax()])
        )
        for i, k in enumerate(cluster_summary.index)
    }


def order_clusters_by_peak(
    psd_clust: pd.DataFrame,
    f: np.ndarray,
    summary: Summary = "mean",
    exclude: list[int] | None = None,
    baseline: np.ndarray | None = None,
    freq_range: tuple[float, float] | None = None,
    feature_cols: Iterable[Hashable] | None = None,
    label_col: str = "clusters",
) -> list[int]:
    """Cluster ids ordered by ascending peak frequency.

    Same ``baseline`` / ``freq_range`` semantics as
    ``cluster_peak_frequencies``. Pass ``exclude=[no_peak_id]`` to drop the
    no-peak cluster (it has no meaningful peak).
    """
    excluded = set(exclude or [])
    peaks = cluster_peak_frequencies(
        psd_clust, f, summary=summary,
        baseline=baseline, freq_range=freq_range,
        feature_cols=feature_cols, label_col=label_col,
        no_peak=excluded,
    )
    keyed = [(k, v) for k, v in peaks.items() if k not in excluded and v is not None]
    return [k for k, _ in sorted(keyed, key=lambda kv: kv[1])]


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