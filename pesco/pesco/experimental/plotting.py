"""Stage 4 of the Frauscher 2018 pipeline: plots, each independently callable."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib import collections as mc
from matplotlib.axes import Axes

from pesco.experimental.clustering import Summary, get_no_peak
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
):
    """Plot per-cluster summary PSD, highlighting the no-peak cluster."""
    smal = smal or []
    cluster_summary = psd_clust.groupby("clusters").agg(summary)
    matplotlib.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    counts = psd_clust["clusters"].value_counts()
    for k in range(len(cluster_summary.index)):
        row = cluster_summary.loc[k]
        label = f"cl. {k} ({counts.loc[k]} el.)"
        if k in smal and k == nopeak:
            ax.semilogx(f, row, linewidth=4.0, color="black", label=label)
        elif k == nopeak:
            ax.semilogx(f, row, linestyle=":", linewidth=5.0, label=label)
        else:
            ax.semilogx(f, row, alpha=0.5, label=label)
    ax.legend()
    ax.set_xticks([0.5, 4, 8, 13, 30, 80])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid()
    for t, text in zip(
        [2, 6, 10, 16, 36],
        [r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma$"],
    ):
        ax.text(t, 0.97, text, fontsize=14,
                transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Normalized spectral density")
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