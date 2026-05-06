"""Stage 4 of the Frauscher 2018 pipeline: plots, each independently callable."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib import collections as mc
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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


def _resolve_feature_cols(
    df: pd.DataFrame, feature_cols: Iterable[Hashable] | None,
) -> list[Hashable]:
    """Return explicit feature columns, or auto-pick numeric-named ones."""
    if feature_cols is not None:
        return list(feature_cols)
    return [c for c in df.columns if isinstance(c, (int, float, np.floating))]


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
    tick_labelsize: float = 10.0,
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
    ax.tick_params(axis="both", labelsize=tick_labelsize)

    summary_curve = channel_data.agg(summary, axis=0).to_numpy(dtype=float)
    q75_curve = channel_data.quantile(0.75, axis=0).to_numpy(dtype=float)
    candidates = []
    for arr in (summary_curve, q75_curve):
        if arr.size:
            candidates.append(float(np.nanmax(arr)))
    if no_peak_center is not None and np.size(no_peak_center) > 0:
        candidates.append(float(np.nanmax(no_peak_center)))
    ymax = max(candidates) * 1.15 if candidates else 0.10
    ax.set_ylim(0, ymax)
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
    ax: "plt.Axes | None" = None,
    show: bool = True,
    save: bool = True,
    title: str | None = None,
    legend_fontsize: float | None = None,
) -> tuple[Figure, Axes]:
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
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure
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
    legend_kwargs = {}
    if legend_fontsize is not None:
        legend_kwargs["fontsize"] = legend_fontsize
    ax.legend(**legend_kwargs)
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
    resolved_title = title if title is not None else (
        f"{summary.capitalize()} power of different PSDs clusters of {dataset}"
    )
    ax.set_title(resolved_title)
    if save and owns_fig:
        output_dir.mkdir(exist_ok=True)
        fig.savefig(output_dir / f"{dataset}_clusters.svg", format="svg")
    if show and owns_fig:
        plt.show()
    return fig, ax


def plot_clusters_pair(
    top: dict,
    bottom: dict,
    *,
    suptitle: str | None = None,
    figsize: tuple[float, float] = (12, 14),
    output_path: Path | str | None = None,
    show: bool = True,
    dpi: int = 300,
    suptitle_fontsize: float = 18,
    suptitle_fontweight: str = "bold",
    sharex: bool = True,
    legend_fontsize: float = 11,
) -> tuple[Figure, np.ndarray]:
    """Stack two ``plot_clusters`` panels vertically (top/bottom).

    Each of ``top`` and ``bottom`` is a kwargs dict forwarded to
    :func:`plot_clusters`. Required: ``psd_clust``, ``f``, ``dataset``.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=sharex)
    for panel_kwargs, ax in ((top, axes[0]), (bottom, axes[1])):
        kw = dict(panel_kwargs)
        kw.setdefault("legend_fontsize", legend_fontsize)
        plot_clusters(ax=ax, show=False, save=False, **kw)

    if suptitle is not None:
        fig.suptitle(
            suptitle, fontsize=suptitle_fontsize, fontweight=suptitle_fontweight,
        )
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    return fig, axes


def plot_histogram(
    psd: pd.DataFrame,
    colbin: pd.Categorical,
    dataset: str,
    cols_to_drop: list[str] | None = None,
    summary: Summary = "mean",
    feature_cols: Iterable[Hashable] | None = None,
    output_path: Path | str | None = None,
) -> tuple[Figure, list[Axes]]:
    """Histogram of summary power per frequency interval (Fig. 4 inset style).

    Two panels: absolute share, and share normalized by interval width.
    Tolerates extra metadata columns in ``psd`` (Region name, Lobe,
    clusters, mni_x/y/z, hemisphere, dataset, region_number, ...) by
    selecting only numeric-named spectral columns. Pass ``feature_cols``
    explicitly to override.
    """
    if feature_cols is None and cols_to_drop:
        cols_present = [c for c in cols_to_drop if c in psd.columns]
        if cols_present:
            psd = psd.drop(cols_present, axis=1)
    psd = psd[_resolve_feature_cols(psd, feature_cols)]

    psd_summary = pd.DataFrame(psd.agg(summary, axis=0)).T
    psd_intervals = get_intervals(psd_summary, colbin)

    fig, ax_arr = plt.subplots(1, 2, sharex=False, figsize=(30, 15))
    ax: list[Axes] = list(ax_arr.flatten())
    matplotlib.rcParams.update({"font.size": 22})

    cats = colbin.categories
    bins = np.array([cats[0].left] + [iv.right for iv in cats], dtype=float)
    bins[0] = bins[0] / 2  # cosmetic: widen leftmost bar on log-x axis

    cat_keys = [str(cat) for cat in cats]
    row = psd_intervals.iloc[0]
    if "Lobe" in row.index:
        row = row.drop(["Lobe"])
    freqs = row.reindex(cat_keys, fill_value=0.0).to_numpy(dtype=float)

    fig.suptitle(dataset)

    ax[0].fill_between(
        bins.repeat(2)[1:-1], freqs.astype(float).repeat(2), facecolor="steelblue"
    )
    ax[0].hlines(y=0.04, xmin=0.5, xmax=80, linewidth=2, color="r")
    ax[0].set_title("absolute share of power in a given bin")
    ax[0].set_xscale("log")
    ax[0].set_xlim(0.5, 80)
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
    ax[1].set_xlim(0.5, 80)
    ax[1].set_ylim(0, 0.07)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_lobes(
    psd_clust: pd.DataFrame,
    psd: pd.DataFrame,
    f: np.ndarray,
    smal: list[int],
    dataset: str,
    sig_lobes: dict[str, list] | None = None,
    summary: Summary = "mean",
    feature_cols: Iterable[Hashable] | None = None,
    tick_labelsize: float = 10.0,
    output_path: Path | str | None = None,
    subplot_height_ratio: float = 1.0,
    font_size: float = 22.0,
) -> tuple[Figure, list[Axes]]:
    """Per-lobe PSD subplots vs no-peak centre, with significance overlays.

    ``subplot_height_ratio`` scales the y-extent of each subplot relative to
    its x-extent. ``1.0`` keeps the original square panels; ``0.6`` makes
    panels shorter (wider-looking).

    For every lobe present in ``psd["Lobe"]``, draws the lobe's per-channel
    summary spectrum (mean/median), IQR band, and min/max envelope on a
    log-frequency axis, overlaid against the no-peak cluster centre derived
    from ``psd_clust``/``smal``. Frequency intervals flagged in
    ``sig_lobes`` are rendered as horizontal segments. Subplot grid sizes
    automatically to the number of lobes (2 cols × ceil(n/2) rows).
    """
    psd_cols = _resolve_feature_cols(psd, feature_cols)
    clust_cols = set(psd_clust.columns)
    cols = [c for c in psd_cols if c in clust_cols]
    f_axis = np.asarray(cols, dtype=float) if len(cols) != len(f) else f
    _, center = get_no_peak(psd_clust, smal, summary=summary, feature_cols=cols)
    matplotlib.rcParams.update({"font.size": font_size})

    canonical = ["Occipital", "Parietal", "Frontal", "Temporal"]
    present = list(psd["Lobe"].dropna().unique())
    lobes = [lb for lb in canonical if lb in present] + [
        lb for lb in present if lb not in canonical
    ]
    n_cols = 2
    n_rows = _ceildiv(len(lobes), n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8 * n_cols, 8 * n_rows * subplot_height_ratio),
        squeeze=False,
    )
    fig.suptitle(f"Lobar differences in EEG frequencies: {dataset}", y=0.98)
    fig.subplots_adjust(top=0.94, hspace=0.25, wspace=0.25)

    flat_axes: list[Axes] = list(axes.flatten())
    for ax, lobe in zip(flat_axes, lobes):
        lobe_psd = psd.loc[psd["Lobe"] == lobe, cols]
        intervals = sig_lobes.get(lobe) if sig_lobes else None
        title = f"{lobe} lobe - {len(lobe_psd)} channels"
        _plot_subplot(
            lobe_psd, center, f_axis, intervals, title,
            ax=ax, summary=summary, tick_labelsize=tick_labelsize,
        )
    for ax in flat_axes[len(lobes):]:
        ax.set_visible(False)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig, flat_axes


def plot_regions(
    psd_clust: pd.DataFrame,
    psd: pd.DataFrame,
    f: np.ndarray,
    smal: list[int],
    dataset: str,
    sig_regions: dict[str, dict[str, list]] | None = None,
    summary: Summary = "mean",
    feature_cols: Iterable[Hashable] | None = None,
    tick_labelsize: float = 8.0,
    output_path_template: str | None = None,
) -> dict[str, Figure]:
    """One figure per lobe, with one subplot per region.

    Returns ``{lobe: fig}``. Use ``fig.axes`` if you need the per-region
    axes. Pass ``output_path_template`` (a format string with ``{lobe}``
    placeholder, e.g. ``"images/{lobe}_regions.svg"``) to save each
    figure; otherwise figures are returned unsaved.
    """
    matplotlib.rcParams.update({"font.size": 8})
    psd_cols = _resolve_feature_cols(psd, feature_cols)
    clust_cols = set(psd_clust.columns)
    cols = [c for c in psd_cols if c in clust_cols]
    f_axis = np.asarray(cols, dtype=float) if len(cols) != len(f) else f
    _, center = get_no_peak(psd_clust, smal, summary=summary, feature_cols=cols)

    figs: dict[str, Figure] = {}
    for lobe in psd["Lobe"].dropna().unique():
        regions = psd.loc[psd["Lobe"] == lobe, "Region name"].unique()
        n_rows = _ceildiv(len(regions), 2)
        fig, axes = plt.subplots(n_rows, 2, figsize=(10, 8 * _ceildiv(n_rows, 2)))
        fig.subplots_adjust(hspace=0.25, wspace=0.25)

        for ax, region in zip(axes.flatten(), regions):
            region_psd = psd.loc[
                (psd["Lobe"] == lobe) & (psd["Region name"] == region), cols
            ]
            intervals = (
                sig_regions[lobe].get(region) if sig_regions else None
            )
            title = f"{_strip_quotes(region)} - {len(region_psd)} channels"
            _plot_subplot(
                region_psd, center, f_axis, intervals, title,
                ax=ax, summary=summary, tick_labelsize=tick_labelsize,
            )

        fig.get_axes()[0].annotate(
            f"Regional differences in {lobe.lower()} lobe ({dataset})",
            (0.5, 0.95), xycoords="figure fraction", ha="center", fontsize=24,
        )

        if output_path_template is not None:
            out = Path(output_path_template.format(lobe=lobe, dataset=dataset))
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, bbox_inches="tight", pad_inches=0)
        figs[lobe] = fig
    return figs


_LOBE_ORDER = ("Occipital", "Parietal", "Frontal", "Temporal")
_LOBE_COLORS = {
    "Occipital": "red",
    "Parietal": "green",
    "Frontal": "#1f6feb",
    "Temporal": "black",
}


def _strip_quotes(value: object) -> str:
    return str(value).strip().strip("'").strip('"').strip()


def _build_region_rows_meta(
    df: pd.DataFrame,
    lobe_order: Sequence[str],
) -> pd.DataFrame:
    rows_meta = df[
        ["__lobe_display", "__region_display", "is_lobe_row"]
    ].drop_duplicates()

    lobe_rank = {lobe: i for i, lobe in enumerate(lobe_order)}
    fallback_rank = len(lobe_order)
    encounter = {key: i for i, key in enumerate(rows_meta["__region_display"])}

    def _row_sort_key(row):
        return (
            lobe_rank.get(row["__lobe_display"], fallback_rank),
            0 if row["is_lobe_row"] else 1,
            encounter[row["__region_display"]],
        )

    rows_meta = rows_meta.assign(__sort=rows_meta.apply(_row_sort_key, axis=1))
    return rows_meta.sort_values("__sort").reset_index(drop=True)


def _prepare_region_df(
    regional_diff: pd.DataFrame,
    region_col: str,
    lobe_col: str,
) -> pd.DataFrame:
    df = regional_diff.copy()
    df["__region_display"] = df[region_col].map(_strip_quotes)
    df["__lobe_display"] = df[lobe_col].map(_strip_quotes)
    if "is_lobe_row" not in df.columns:
        df["is_lobe_row"] = df["__region_display"] == df["__lobe_display"]
    return df


def plot_region_difference_heatmap(
    regional_diff: pd.DataFrame,
    *,
    title: str | None = None,
    value_col: str = "channel_fraction",
    region_col: str = "Region name",
    lobe_col: str = "Lobe",
    interval_col: str = "interval",
    cmap: str = "YlGnBu",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annotate: bool = True,
    annot_fontsize: float = 7.0,
    output_path: Path | str | None = None,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    lobe_order: Sequence[str] = _LOBE_ORDER,
    reference_df: pd.DataFrame | None = None,
    ax: "plt.Axes | None" = None,
    show_yticks: bool = True,
    show_ylabel: bool = True,
    cbar: bool = True,
    cbar_ax: "plt.Axes | None" = None,
    cbar_kws: dict | None = None,
    facecolor: str | None = None,
    xtick_rotation: float = 90,
    xtick_ha: str | None = None,
    ytick_rotation: float = 0,
    ytick_ha: str | None = None,
    tick_fontsize: float | None = None,
    title_fontsize: float | None = None,
    title_loc: str = "center",
    title_fontweight: str = "normal",
    axis_label_fontsize: float | None = None,
    cbar_label_fontsize: float | None = None,
    cbar_tick_fontsize: float | None = None,
    dpi: int = 300,
    square: bool = False,
):
    """Plot a Frauscher-style regional difference heatmap.

    ``regional_diff`` is the tidy output of
    :func:`pesco.experimental.peak_testing.test_regions_heatmap`. Cells that
    do not pass the corrected KS screen are left blank. Rows are grouped by
    lobe, with optional lobe-aggregate rows (``is_lobe_row``) appearing as
    the first row of each lobe block.

    Pass ``reference_df`` (same schema as ``regional_diff``) to lock the row
    order and lobe metadata to that reference — useful for stacking two
    heatmaps with matching rows. Regions absent from ``regional_diff`` render
    as blank rows.
    """
    import seaborn as sns

    required = {
        region_col,
        lobe_col,
        interval_col,
        value_col,
        "interval_left",
        "ks_significant",
    }
    missing = required.difference(regional_diff.columns)
    if missing:
        raise ValueError(
            "regional_diff is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    df = _prepare_region_df(regional_diff, region_col, lobe_col)

    interval_order = (
        df[[interval_col, "interval_left"]]
        .drop_duplicates()
        .sort_values("interval_left")[interval_col]
        .to_list()
    )

    if reference_df is not None:
        ref_missing = required.difference(reference_df.columns)
        if ref_missing:
            raise ValueError(
                "reference_df is missing required columns: "
                f"{', '.join(sorted(ref_missing))}"
            )
        meta_source = _prepare_region_df(reference_df, region_col, lobe_col)
    else:
        meta_source = df

    rows_meta = _build_region_rows_meta(meta_source, lobe_order)

    region_order = rows_meta["__region_display"].to_list()
    lobe_by_region = dict(
        zip(rows_meta["__region_display"], rows_meta["__lobe_display"])
    )
    is_lobe_row = dict(
        zip(rows_meta["__region_display"], rows_meta["is_lobe_row"])
    )

    values = df.pivot(
        index="__region_display",
        columns=interval_col,
        values=value_col,
    ).reindex(index=region_order, columns=interval_order)
    significant = df.pivot(
        index="__region_display",
        columns=interval_col,
        values="ks_significant",
    ).reindex(index=region_order, columns=interval_order)
    plot_values = values.where(significant)
    mask = plot_values.isna()

    owns_fig = ax is None
    if owns_fig:
        if figsize is None:
            figsize = (
                max(12.0, 0.7 * len(interval_order) + 4.0),
                max(8.0, 0.38 * len(region_order) + 2.0),
            )
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if facecolor is not None:
        ax.set_facecolor(facecolor)

    if annotate:
        labels = plot_values.map(lambda value: "" if pd.isna(value) else f"{value:.2g}")
    else:
        labels = False

    default_cbar_kws = {
        "label": "% of significantly different channels",
        "shrink": 0.55,
        "aspect": 30,
        "pad": 0.02,
    }
    if cbar_kws:
        default_cbar_kws.update(cbar_kws)

    sns.heatmap(
        plot_values,
        mask=mask,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=labels,
        fmt="",
        annot_kws={"size": annot_fontsize},
        linewidths=0.5,
        linecolor="white",
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=default_cbar_kws if cbar else None,
        xticklabels=interval_order,
        yticklabels=region_order if show_yticks else False,
        square=square,
        ax=ax,
    )
    ax.set_yticks(np.arange(len(region_order)) + 0.5)
    ax.set_yticklabels(region_order)
    if not show_yticks:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xticks(np.arange(len(interval_order)) + 0.5)
    ax.set_xticklabels(interval_order)

    if title is not None:
        title_kwargs = {"loc": title_loc, "fontweight": title_fontweight}
        if title_fontsize is not None:
            title_kwargs["fontsize"] = title_fontsize
        ax.set_title(title, **title_kwargs)
    label_kwargs = {}
    if axis_label_fontsize is not None:
        label_kwargs["fontsize"] = axis_label_fontsize
    ax.set_xlabel("Frequencies [Hz]", **label_kwargs)
    ax.set_ylabel(
        "Regions" if (show_yticks and show_ylabel) else "", **label_kwargs
    )
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    if cbar and (cbar_label_fontsize is not None or cbar_tick_fontsize is not None):
        cb = ax.collections[0].colorbar
        if cb is not None:
            if cbar_label_fontsize is not None:
                cb.set_label(
                    default_cbar_kws.get("label", ""),
                    fontsize=cbar_label_fontsize,
                )
            if cbar_tick_fontsize is not None:
                cb.ax.tick_params(labelsize=cbar_tick_fontsize)
    ha = xtick_ha if xtick_ha is not None else ("right" if 0 < xtick_rotation < 90 else "center")
    plt.setp(
        ax.get_xticklabels(),
        rotation=xtick_rotation,
        ha=ha,
        rotation_mode="anchor" if 0 < xtick_rotation < 90 else "default",
    )
    ax.tick_params(axis="x", length=0)
    if show_yticks and ytick_rotation:
        y_ha = ytick_ha if ytick_ha is not None else "right"
        plt.setp(
            ax.get_yticklabels(),
            rotation=ytick_rotation,
            ha=y_ha,
            rotation_mode="anchor",
        )
    ax.tick_params(axis="y", length=0)

    if show_yticks:
        for tick in ax.get_yticklabels():
            name = tick.get_text()
            if is_lobe_row.get(name, False):
                tick.set_color("black")
                tick.set_fontweight("bold")
            else:
                lobe = lobe_by_region.get(name)
                tick.set_color(_LOBE_COLORS.get(lobe, "black"))

    region_lobes = [lobe_by_region[region] for region in region_order]
    for idx, (previous, current) in enumerate(
        zip(region_lobes, region_lobes[1:]), start=1
    ):
        if previous != current:
            ax.axhline(idx, color="white", linewidth=2.0)

    if owns_fig:
        fig.tight_layout()
    if output_path is not None and owns_fig:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show and owns_fig:
        plt.show()
    return fig, ax


def plot_region_difference_heatmap_pair(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    suptitle: str | None = None,
    left_title: str | None = None,
    right_title: str | None = None,
    output_path: Path | str | None = None,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    width_ratios: tuple[float, float] | None = None,
    lobe_order: Sequence[str] = _LOBE_ORDER,
    region_col: str = "Region name",
    lobe_col: str = "Lobe",
    facecolor: str | None = None,
    xtick_rotation: float = 45,
    ytick_rotation: float = 20,
    show_left_ylabel: bool = False,
    tick_fontsize: float = 11,
    title_fontsize: float = 14,
    title_loc: str = "left",
    title_fontweight: str = "bold",
    suptitle_fontsize: float = 16,
    suptitle_fontweight: str = "bold",
    suptitle_y: float = 0.98,
    axis_label_fontsize: float = 12,
    cbar_label_fontsize: float = 11,
    cbar_tick_fontsize: float = 10,
    dpi: int = 300,
    square: bool = False,
    wspace: float | None = None,
    cbar_shrink: float = 0.55,
    **plot_kwargs,
):
    """Side-by-side heatmaps with shared row order.

    Row order/lobe metadata derived from ``pd.concat([left_df, right_df])`` so
    both panels use the same region rows. Right panel has y-tick labels and
    colorbar suppressed (shared via the left panel).
    """
    reference_df = pd.concat([left_df, right_df], ignore_index=True)

    meta_source = _build_region_rows_meta(
        _prepare_region_df(reference_df, region_col, lobe_col), lobe_order
    )
    n_rows = len(meta_source)
    interval_left = (
        left_df[["interval", "interval_left"]]
        .drop_duplicates()
        .sort_values("interval_left")["interval"]
        .to_list()
    )
    interval_right = (
        right_df[["interval", "interval_left"]]
        .drop_duplicates()
        .sort_values("interval_left")["interval"]
        .to_list()
    )
    if figsize is None:
        total_intervals = len(interval_left) + len(interval_right)
        if square:
            cell_size = 0.32
            label_margin = 5.0
            figsize = (
                cell_size * total_intervals + label_margin,
                cell_size * n_rows + 2.5,
            )
        else:
            figsize = (
                max(18.0, 0.55 * total_intervals + 6.0),
                max(8.0, 0.38 * n_rows + 2.0),
            )
    n_left = len(interval_left)
    n_right = len(interval_right)
    if width_ratios is None:
        width_ratios = (float(n_left), float(n_right))

    cbar_width_ratio = 0.4
    gridspec_kw = {
        "width_ratios": list(width_ratios) + [cbar_width_ratio],
    }
    effective_wspace = wspace if wspace is not None else (0.05 if square else None)
    if effective_wspace is not None:
        gridspec_kw["wspace"] = effective_wspace
    fig, all_axes = plt.subplots(
        1, 3, figsize=figsize, gridspec_kw=gridspec_kw,
    )
    axes = all_axes[:2]
    cax = all_axes[2]
    axes[1].sharey(axes[0])

    style_kwargs = dict(
        tick_fontsize=tick_fontsize,
        title_fontsize=title_fontsize,
        title_loc=title_loc,
        title_fontweight=title_fontweight,
        axis_label_fontsize=axis_label_fontsize,
        cbar_label_fontsize=cbar_label_fontsize,
        cbar_tick_fontsize=cbar_tick_fontsize,
        square=False,
    )

    plot_region_difference_heatmap(
        left_df,
        title=left_title,
        reference_df=reference_df,
        ax=axes[0],
        show_yticks=True,
        show_ylabel=show_left_ylabel,
        cbar=False,
        show=False,
        lobe_order=lobe_order,
        region_col=region_col,
        lobe_col=lobe_col,
        facecolor=facecolor,
        xtick_rotation=xtick_rotation,
        ytick_rotation=ytick_rotation,
        **style_kwargs,
        **plot_kwargs,
    )
    plot_region_difference_heatmap(
        right_df,
        title=right_title,
        reference_df=reference_df,
        ax=axes[1],
        show_yticks=False,
        cbar=True,
        cbar_ax=cax,
        show=False,
        lobe_order=lobe_order,
        region_col=region_col,
        lobe_col=lobe_col,
        facecolor=facecolor,
        xtick_rotation=xtick_rotation,
        ytick_rotation=ytick_rotation,
        **style_kwargs,
        **plot_kwargs,
    )

    suptitle_obj = None
    if suptitle is not None:
        suptitle_obj = fig.suptitle(
            suptitle,
            fontsize=suptitle_fontsize,
            fontweight=suptitle_fontweight,
            y=suptitle_y,
        )

    rect = [0, 0, 1, 0.96] if suptitle is not None else None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
        )
        fig.tight_layout(rect=rect)
    if effective_wspace is not None:
        fig.subplots_adjust(wspace=effective_wspace)

    # Equalize geometry: same y0/height on both panels, identical cell width.
    # If square=True, also set height = cell_w * n_rows so cells are square.
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    pos_cax = cax.get_position()

    y0 = max(pos0.y0, pos1.y0)
    height = min(pos0.y0 + pos0.height, pos1.y0 + pos1.height) - y0

    cell_w = min(pos0.width / n_left, pos1.width / n_right)
    if square:
        target_h = cell_w * n_rows
        if target_h <= height:
            top = y0 + height
            y0 = top - target_h
            height = target_h
        else:
            cell_w *= height / target_h

    new_w_left = cell_w * n_left
    new_w_right = cell_w * n_right
    gap = pos1.x0 - (pos0.x0 + pos0.width)
    cax_gap = pos_cax.x0 - (pos1.x0 + pos1.width)

    x_left = pos0.x0
    x_right = x_left + new_w_left + gap
    x_cax = x_right + new_w_right + cax_gap

    axes[0].set_position([x_left, y0, new_w_left, height])
    axes[1].set_position([x_right, y0, new_w_right, height])
    cbar_h = height * cbar_shrink
    cbar_y0 = y0 + (height - cbar_h) / 2
    cax.set_position([x_cax, cbar_y0, pos_cax.width, cbar_h])

    if suptitle_obj is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_tops = [
            ax.title.get_window_extent(renderer).y1
            for ax in axes
            if ax.get_title()
        ]
        if title_tops:
            top_fig = max(title_tops) / fig.bbox.height
            sup_bbox = suptitle_obj.get_window_extent(renderer)
            sup_h_fig = (sup_bbox.y1 - sup_bbox.y0) / fig.bbox.height
            suptitle_obj.set_y(top_fig + sup_h_fig / 2 + 0.005)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    return fig, axes


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
