"""Stage 4 of the Frauscher 2018 pipeline: plots, each independently callable."""

from __future__ import annotations

from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib import collections as mc
from matplotlib.axes import Axes

from pesco.experimental.clustering import (
    EEG_BANDS,
    Summary,
    band_edges,
    cluster_bands,
    cluster_peak_frequencies,
    get_no_peak,
    order_clusters_by_peak,
)
from pesco.experimental.peak_testing import get_intervals


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------

def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def _plot_subplot(
    channel_data: pd.DataFrame,
    no_peak_center: np.ndarray | None,
    f: np.ndarray,
    sig_intervals: list | None,
    title: str,
    ax: Axes | None = None,
    summary: Summary = "mean",
) -> Axes:
    """Plot one PSD subplot: summary + IQR + min/max + significance overlay.

    If no_peak_center is None, the no-peak reference line is omitted.
    """
    if ax is None:
        ax = plt.gca()

    q75 = channel_data.quantile(0.75, axis=0)
    q25 = channel_data.quantile(0.25, axis=0)

    if no_peak_center is not None:
        ax.semilogx(f, no_peak_center, color="black")
    ax.semilogx(f, q75, color="pink")
    ax.semilogx(f, q25, color="pink")
    ax.semilogx(f, channel_data.agg(summary, axis=0), color="red")
    ax.semilogx(f, channel_data.max(axis=0), color="r", linestyle=":")
    ax.semilogx(f, channel_data.min(axis=0), color="r", linestyle=":")
    ax.fill_between(f, q25, q75, facecolor="pink", interpolate=True)

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid()

    if sig_intervals:
        lc = mc.LineCollection(sig_intervals, linewidths=2)
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Normalized spectral density")
    ax.set_title(title)
    for t, text in zip(
        [2, 6, 10, 16, 36],
        [r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma$"],
    ):
        ax.text(t, 0.97, text, fontsize=14,
                transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.set_xticks([0.5, 4, 8, 13, 30, 80])
    ax.set_ylim(0, 0.10)
    ax.set_xlim(0.5, 80)
    return ax


# ---------------------------------------------------------------------------
# public plotting functions
# ---------------------------------------------------------------------------

def plot_single_psd(psd_df: pd.DataFrame, f: np.ndarray, i: int) -> None:
    """Plot the PSD of a single channel (row i)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.semilogx(f, psd_df.iloc[i])
    ax.grid()
    ax.set_xlim(0.5, 80)
    ax.set_xticks([0.5, 4, 8, 13, 30, 80])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Frequency")
    ax.set_ylabel("PSD")


def plot_clusters(
    psd_clust: pd.DataFrame,
    f: np.ndarray,
    dataset: str,
    smal: list[int] | None = None,
    nopeak: int | None = None,
    output_dir: Path = Path("images"),
    summary: Summary = "mean",
    log_y: bool = False,
    order_by_peak: bool = False,
    cmap: str = "viridis",
    peak_baseline: np.ndarray | None = None,
    peak_freq_range: tuple[float, float] | None = None,
    feature_cols: Iterable[Hashable] | None = None,
    label_col: str = "clusters",
    label_by_band: bool = False,
    ylabel: str | None = None,
):
    """Plot per-cluster summary PSD, highlighting the no-peak cluster.

    Set log_y=True to use a logarithmic scale on the PSD axis.
    Set order_by_peak=True to colour clusters by ascending peak frequency
    (no-peak cluster stays black/darkest, low-freq peaks dark, high-freq
    peaks lighter along ``cmap``).

    ``peak_baseline`` / ``peak_freq_range`` are forwarded to
    ``order_clusters_by_peak`` so the peak is taken relative to a baseline
    (e.g. the no-peak cluster summary) and/or restricted to a band — this
    picks the genuine spectral peak rather than the δ/1f maximum.
    """
    smal = smal or []
    if feature_cols is None:
        feature_cols = [
            c for c in psd_clust.columns
            if c != label_col and isinstance(c, (int, float, np.floating))
        ]
    feature_cols = list(feature_cols)
    cluster_summary = (
        psd_clust[feature_cols].groupby(psd_clust[label_col]).agg(summary)
    )
    matplotlib.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    counts = psd_clust[label_col].value_counts()

    band_labels: dict[int, str] = {}
    if label_by_band:
        peaks = cluster_peak_frequencies(
            psd_clust, f, summary=summary,
            baseline=peak_baseline, freq_range=peak_freq_range,
            feature_cols=feature_cols, label_col=label_col,
            no_peak=smal,
        )
        band_labels = cluster_bands(peaks)

    if order_by_peak:
        sorted_keys = order_clusters_by_peak(
            psd_clust, f, summary=summary,
            exclude=[nopeak] if nopeak is not None else None,
            baseline=peak_baseline,
            freq_range=peak_freq_range,
            feature_cols=feature_cols,
            label_col=label_col,
        )
        cm = plt.get_cmap(cmap)
        n = len(sorted_keys)
        color_map = {k: cm(i / max(n - 1, 1)) for i, k in enumerate(sorted_keys)}
        plot_order = sorted_keys + (
            [nopeak] if nopeak is not None and nopeak in cluster_summary.index else []
        )
    else:
        color_map = {}
        plot_order = list(cluster_summary.index)

    for k in plot_order:
        row = cluster_summary.loc[k]
        if label_by_band and k in band_labels:
            label = f"{band_labels[k]} ({counts.loc[k]})"
        else:
            label = f"cl. {k} ({counts.loc[k]} el.)"
        if k in smal and k == nopeak:
            ax.semilogx(f, row, linewidth=4.0, color="black", label=label)
        elif k == nopeak:
            ax.semilogx(f, row, linestyle=":", linewidth=5.0, color="black", label=label)
        elif k in color_map:
            ax.semilogx(f, row, color=color_map[k], linewidth=2.0, label=label)
        else:
            ax.semilogx(f, row, alpha=0.5, label=label)
    ax.legend()
    ax.set_xticks(band_edges())
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if log_y:
        ax.set_yscale("log")
    ax.grid()
    for b in EEG_BANDS:
        ax.text(
            b.label_x, 0.97, f"${b.tex}$", fontsize=14,
            transform=ax.get_xaxis_transform(), ha="center", va="top",
        )
    ax.set_xlabel("Frequency")
    if ylabel is None:
        ylabel = (
            "Log of normalized spectral density"
            if log_y else "Normalized spectral density"
        )
    ax.set_ylabel(ylabel)
    plt.title(f"{summary.capitalize()} power of different PSDs clusters of {dataset}")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{dataset}_clusters.svg", format="svg")
    plt.show()
    return fig, ax


def plot_histogram(
    psd: pd.DataFrame,
    colbin: pd.Categorical,
    dataset: str,
    cols_to_drop: list[str] | None = None,
    output_dir: Path = Path("images"),
    summary: Summary = "mean",
) -> None:
    """Histogram of summary power per frequency interval (Fig. 4 inset style).

    Two panels: absolute share, and share normalized by interval width.
    """
    if cols_to_drop is None:
        cols_to_drop = ["Region name", "Lobe"]
    cols_present = [c for c in cols_to_drop if c in psd.columns]
    if cols_present:
        psd = psd.drop(cols_present, axis=1)

    psd_summary = pd.DataFrame(psd.agg(summary, axis=0)).T
    psd_intervals = get_intervals(psd_summary, colbin)

    fig, ax = plt.subplots(1, 2, sharex=False, figsize=(30, 15))
    ax = ax.flatten()
    matplotlib.rcParams.update({"font.size": 22})

    cats = colbin.categories
    bins = np.array([cats[0].left] + [iv.right for iv in cats], dtype=float)
    bins[0] = bins[0] / 2  # cosmetic: widen leftmost bar on log-x axis

    if "Lobe" in psd_intervals.columns:
        freqs = psd_intervals.iloc[0].drop(["Lobe"]).values
    else:
        freqs = psd_intervals.iloc[0].values

    fig.suptitle(dataset)

    ax[0].fill_between(
        bins.repeat(2)[1:-1], freqs.astype(float).repeat(2), facecolor="steelblue"
    )
    ax[0].hlines(y=0.04, xmin=0.5, xmax=80, linewidth=2, color="r")
    ax[0].set_title("absolute share of power in a given bin")
    ax[0].set_xscale("log")
    ax[0].set_xlim([0.5, 80])
    ax[0].set_xticks([0.5, 4, 8, 13, 30, 80])
    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0].grid()
    ax[0].set_ylim(0, 0.40)

    widths = bins[1:] - bins[:-1]
    heights = freqs.astype(float) / widths
    ax[1].fill_between(bins.repeat(2)[1:-1], heights.repeat(2), facecolor="steelblue")
    ax[1].set_title("share of power in a given bin normalized by width of the bin")
    ax[1].set_xscale("log")
    ax[1].set_xticks([0.5, 4, 8, 13, 30, 80])
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].grid()
    ax[1].set_xlim([0.5, 80])
    ax[1].set_ylim(0, 0.07)

    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "mean_power_share_in_intervals.png", format="png")
    plt.show()


def plot_lobes(
    psd_clust: pd.DataFrame,
    psd: pd.DataFrame,
    f: np.ndarray,
    smal: list[int],
    dataset: str,
    sig_lobes: dict[str, list] | None = None,
    show: bool = False,
    output_dir: Path = Path("images"),
    summary: Summary = "mean",
) -> None:
    """4-panel plot: PSD per lobe with significant-interval overlays."""
    _, center = get_no_peak(psd_clust, smal, summary=summary)
    matplotlib.rcParams.update({"font.size": 22})

    lobes = ["Occipital", "Parietal", "Frontal", "Temporal"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(f"Lobar differences in EEG frequencies: {dataset}")
    fig.subplots_adjust(hspace=0.25, wspace=0.25)

    for ax, lobe in zip(axes.flatten(), lobes):
        lobe_psd = psd[psd.Lobe == lobe].drop(["Region name", "Lobe"], axis=1)
        intervals = sig_lobes[lobe] if sig_lobes else None
        title = f"{lobe} lobe - {len(lobe_psd)} channels"
        _plot_subplot(lobe_psd, center, f, intervals, title, ax=ax, summary=summary)

    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{dataset}_lobar_differences.svg", format="svg")
    if show:
        plt.show()
    else:
        plt.close()


def plot_regions(
    psd_clust: pd.DataFrame,
    psd: pd.DataFrame,
    f: np.ndarray,
    smal: list[int],
    dataset: str,
    sig_regions: dict[str, dict[str, list]] | None = None,
    output_dir: Path = Path("images"),
    summary: Summary = "mean",
) -> None:
    """One figure per lobe, with one subplot per region."""
    matplotlib.rcParams.update({"font.size": 8})
    _, center = get_no_peak(psd_clust, smal, summary=summary)
    output_dir.mkdir(exist_ok=True)

    for lobe in psd["Lobe"].unique():
        regions = psd[psd.Lobe == lobe]["Region name"].unique()
        n_rows = _ceildiv(len(regions), 2)
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 8 * _ceildiv(n_rows, 2)))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)

        for ax, region in zip(axes.flatten(), regions):
            region_psd = psd[
                (psd.Lobe == lobe) & (psd["Region name"] == region)
            ].drop(["Region name", "Lobe"], axis=1)
            intervals = (
                sig_regions[lobe].get(region) if sig_regions else None
            )
            title = f"{region} - {len(region_psd)} channels"
            _plot_subplot(region_psd, center, f, intervals, title, ax=ax, summary=summary)

        fig.get_axes()[0].annotate(
            f"Regional differences in {lobe.lower()} lobe ({dataset})",
            (0.5, 0.95), xycoords="figure fraction", ha="center", fontsize=24,
        )
        plt.savefig(
            output_dir / f"{dataset}_{lobe}_regional_differences.svg",
            format="svg", bbox_inches="tight", pad_inches=0,
        )
        plt.show()


# ---------------------------------------------------------------------------
# Frauscher-style brain plot: electrodes coloured by cluster on glass brain
# ---------------------------------------------------------------------------

def plot_cluster_brain(
    positions: np.ndarray,
    labels: np.ndarray | Sequence,
    *,
    label_order: Sequence | None = None,
    label_colors: Mapping | None = None,
    label_names: Mapping | None = None,
    label_markers: Mapping | None = None,
    default_marker: str = "o",
    cmap: str = "viridis",
    display_mode: str = "lzry",
    node_size: float = 30.0,
    alpha: float = 0.7,
    title: str | None = None,
    output_path: Path | str | None = None,
    show: bool = True,
    legend_loc: str = "center left",
    legend_bbox: tuple[float, float] = (-0.02, 0.5),
    legend_fontsize: float = 9,
    **plot_markers_kwargs,
):
    """Plot electrodes on a glass brain, coloured by cluster (Frauscher-style).

    Thin wrapper around ``nilearn.plotting.plot_markers``.

    Parameters
    ----------
    positions : (n, 3) MNI electrode coords (e.g. ``ChannelPosition``).
    labels : (n,) cluster id / band per electrode. NaN or None entries skipped.
    label_order : legend order; defaults to sorted unique labels.
    label_colors : {label: matplotlib colour}; defaults to ``cmap`` sampled in
        ``label_order``.
    label_names : {label: display name} for legend.
    label_markers : {label: marker}; labels omitted here use ``default_marker``.
        Use e.g. ``{no_peak_label: "^"}`` to draw no-peak clusters as triangles.
    default_marker : matplotlib marker for labels not listed in ``label_markers``.
    display_mode : nilearn glass-brain mode. ``"lzry"`` = left/coronal/axial/right.
    node_size : marker size in pt^2.
    output_path : save figure to this path if given.
    """
    from nilearn import plotting as nlp
    from matplotlib.colors import ListedColormap

    positions = np.asarray(positions, dtype=float)
    labels = np.asarray(labels, dtype=object)
    valid = np.array([
        label is not None
        and not (isinstance(label, float) and np.isnan(label))
        for label in labels
    ])
    positions = positions[valid]
    labels = labels[valid]
    if not isinstance(node_size, str) and np.ndim(node_size) > 0:
        node_size = np.asarray(node_size)
        if len(node_size) == len(valid):
            node_size = node_size[valid]

    if label_order is None:
        try:
            label_order = sorted(set(labels))
        except TypeError:
            label_order = list(dict.fromkeys(labels))
    label_order = list(label_order)

    if label_colors is None:
        cm = plt.get_cmap(cmap)
        n = len(label_order)
        label_colors = {
            label: cm(i / max(n - 1, 1))
            for i, label in enumerate(label_order)
        }

    label_to_int = {label: i for i, label in enumerate(label_order)}
    node_values = np.array(
        [label_to_int[label] for label in labels], dtype=float
    )
    colors = [label_colors[label] for label in label_order]
    listed = ListedColormap(colors)

    label_markers = dict(label_markers or {})
    plot_markers_kwargs = dict(plot_markers_kwargs)
    node_kwargs = dict(plot_markers_kwargs.pop("node_kwargs", {}) or {})
    default_marker = node_kwargs.get("marker", default_marker)
    marker_for_label = {
        label: label_markers.get(label, default_marker)
        for label in label_order
    }

    def _layer_size(mask: np.ndarray):
        if isinstance(node_size, str):
            if node_size == "auto":
                return min(1e4 / len(positions), 100)
            return node_size
        if np.ndim(node_size) == 0:
            return node_size
        return np.asarray(node_size)[mask]

    unique_markers = list(dict.fromkeys(marker_for_label.values()))
    base_marker = (
        default_marker
        if default_marker in unique_markers
        else unique_markers[0]
    )
    base_mask = np.array(
        [
            marker_for_label.get(label, default_marker) == base_marker
            for label in labels
        ]
    )
    base_node_kwargs = dict(node_kwargs)
    base_node_kwargs["marker"] = base_marker

    display = nlp.plot_markers(
        node_values=node_values[base_mask],
        node_coords=positions[base_mask],
        node_cmap=listed,
        node_vmin=-0.5, node_vmax=len(label_order) - 0.5,
        node_size=_layer_size(base_mask),
        alpha=alpha,
        display_mode=display_mode,
        colorbar=False,
        title=title,
        node_kwargs=base_node_kwargs,
        **plot_markers_kwargs,
    )

    for marker in unique_markers:
        if marker == base_marker:
            continue
        marker_mask = np.array(
            [
                marker_for_label.get(label, default_marker) == marker
                for label in labels
            ]
        )
        display.add_markers(
            marker_coords=positions[marker_mask],
            marker_color=[
                label_colors[label] for label in labels[marker_mask]
            ],
            marker_size=_layer_size(marker_mask),
            marker=marker,
            alpha=alpha,
        )

    counts = {
        label: int(np.sum(labels == label)) for label in label_order
    }
    legend_handles = [
        plt.Line2D(
            [0], [0], marker=marker_for_label[label], linestyle="",
            markerfacecolor=label_colors[label], markeredgecolor="black",
            markersize=8,
            label=f"{(label_names or {}).get(label, str(label))} "
                  f"({counts[label]})",
        )
        for label in label_order
    ]
    fig = plt.gcf()
    fig.legend(
        handles=legend_handles, loc=legend_loc,
        bbox_to_anchor=legend_bbox, frameon=True, fontsize=legend_fontsize,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return display
