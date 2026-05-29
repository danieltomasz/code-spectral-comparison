"""Relative band-power normative maps and cross-modal correlation.

Reproduces the Janiukstyte et al. 2023 (Normative Brain Mapping) analysis:
compute relative power in canonical EEG bands for every channel, collapse
channels into a shared spatial unit (ROI/region), then correlate the resulting
normative maps across two modalities (e.g. source HD-EEG vs intracranial EEG).

The pipeline is four stages, each a pure function returning a tidy DataFrame
(or a Matplotlib figure). Per the project convention, plotting returns the
figure only; the caller is responsible for ``savefig``.

    1. relative_band_power_by_channel  -- PSD table  -> one row per (channel, band)
    2. region_band_power               -- collapse channels into ROI maps
    3. compare_region_band_power       -- pair the two modalities' maps, Spearman rho
    4. plot_band_power_correlation_grid-- one scatter panel per band

Usage
-----
>>> from pesco.bandpower import (
...     relative_band_power_by_channel,
...     region_band_power,
...     compare_region_band_power,
...     plot_band_power_correlation_grid,
... )
>>> # psd_df: rows = channels, float columns = frequencies, plus metadata
>>> # columns ("Region name", "Lobe", "mni_x", ...). One per modality.
>>> ieeg = relative_band_power_by_channel(ieeg_psd_df, ieeg_f, dataset="iEEG")
>>> hd   = relative_band_power_by_channel(hd_psd_df,   hd_f,   dataset="HD-EEG")
>>> by_channel = pd.concat([ieeg, hd], ignore_index=True)
>>> regions = region_band_power(by_channel, summary="mean", roi_level="bilateral")
>>> paired, correlations = compare_region_band_power(
...     regions, x_dataset="HD-EEG", y_dataset="iEEG", roi_col="roi_bilateral"
... )
>>> fig = plot_band_power_correlation_grid(paired, correlations)  # color_by="band"
>>> fig.savefig("bandpower_correlations.svg", bbox_inches="tight")

Band definitions are reused from :data:`pesco.experimental.clustering.EEG_BANDS`
(the project's single source of truth for canonical band edges). Display names
and per-band colors live in :data:`BAND_DISPLAY`.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats

from pesco.experimental.clustering import Band, EEG_BANDS, Summary
from pesco.preprocess import normalize_psd

ColorBy = Literal["band", "lobe"]

# Display name (long form) and scatter color for each band, keyed by the band's
# short ``name`` (the Greek symbol in ``EEG_BANDS``). Colors follow Janiukstyte
# Fig. 3A. To match the figure's colored panel titles, the rho title is drawn
# in the band's color when ``color_by="band"``.
BAND_DISPLAY: dict[str, tuple[str, str]] = {
    "δ": ("delta", "#4C9BE8"),
    "θ": ("theta", "#B267E6"),
    "α": ("alpha", "#E65DCB"),
    "β": ("beta", "#7AC45C"),
    "γ": ("gamma", "#69D0D3"),
}
BAND_LABELS: dict[str, str] = {name: long for name, (long, _) in BAND_DISPLAY.items()}
BAND_COLORS: dict[str, str] = {name: color for name, (_, color) in BAND_DISPLAY.items()}

# Anatomical-lobe coloring, used only when ``color_by="lobe"``.
LOBE_ORDER: tuple[str, ...] = ("Occipital", "Parietal", "Frontal", "Temporal", "Insula")
LOBE_COLORS: dict[str, str] = {
    "Occipital": "red",
    "Parietal": "green",
    "Frontal": "#1f6feb",
    "Temporal": "black",
    "Insula": "black",
}


def metadata_columns(df: pd.DataFrame) -> list:
    """Return the string-named (metadata) columns, excluding numeric freq columns."""
    return [c for c in df.columns if isinstance(c, str)]


def relative_psd_df(psd_df: pd.DataFrame, freqs: np.ndarray) -> pd.DataFrame:
    """Return PSD normalized to unit in-band power, preserving metadata.

    Parameters
    ----------
    psd_df : DataFrame
        Rows are channels; float-named columns are frequencies; string-named
        columns are metadata.
    freqs : array, shape (F,)
        Frequency vector identifying the PSD columns.

    Returns
    -------
    DataFrame
        Same shape as ``psd_df``; the frequency columns are normalized per row
        via :func:`pesco.preprocess.normalize_psd`, metadata columns unchanged.
    """
    freqs = np.asarray(freqs, dtype=float)
    rel = normalize_psd(psd_df[list(freqs)].to_numpy(dtype=float), freqs)
    return pd.DataFrame(rel, index=psd_df.index, columns=freqs).join(
        psd_df[metadata_columns(psd_df)]
    )


def add_roi_hemisphere_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add hemisphere-specific and bilateral ROI labels for map comparison.

    Adds three columns:

    - ``hemisphere`` -- "L"/"R" from the sign of ``mni_x`` if absent. Left
      untouched when already present (e.g. derived from channel names).
    - ``roi_hemisphere`` -- ``"<Region name> (<hemisphere>)"``, the unit for
      the hemisphere-specific comparison.
    - ``roi_bilateral`` -- ``Region name``, pooling left/right channels.
    """
    metadata = metadata.copy()
    if "hemisphere" not in metadata.columns:
        if "mni_x" not in metadata.columns:
            metadata["hemisphere"] = pd.NA
        else:
            metadata["hemisphere"] = np.select(
                [metadata["mni_x"] < 0, metadata["mni_x"] > 0],
                ["L", "R"],
                default=pd.NA,
            )
    metadata["hemisphere"] = metadata["hemisphere"].astype("string")
    metadata["roi_hemisphere"] = (
        metadata["Region name"].astype(str) + " (" + metadata["hemisphere"] + ")"
    )
    metadata["roi_bilateral"] = metadata["Region name"].astype(str)
    return metadata


def relative_band_power_by_channel(
    psd_df: pd.DataFrame,
    freqs: np.ndarray,
    dataset: str,
    bands: Iterable[Band] = EEG_BANDS,
) -> pd.DataFrame:
    """Integrate the (uncorrected) PSD into relative band power per channel.

    For each channel, band power is the PSD integral over the band, and
    relative power is band power divided by the channel's total in-band power.
    Bands are half-open ``[lo, hi)`` except the final band, which is closed
    ``[lo, hi]`` so the top edge is included.

    Parameters
    ----------
    psd_df : DataFrame
        Rows are channels; float-named columns are frequencies. Recognized
        metadata columns ("Region name", "Lobe", "hemisphere", "mni_x") are
        carried through and used for ROI labeling.
    freqs : array, shape (F,)
        Frequency vector identifying the PSD columns.
    dataset : str
        Modality label written to the ``dataset`` column (used later to pair
        the two maps in :func:`compare_region_band_power`).
    bands : iterable of Band, optional
        Defaults to :data:`pesco.experimental.clustering.EEG_BANDS`.

    Returns
    -------
    DataFrame
        Tidy, one row per (channel, band), with columns ``channel``,
        ``dataset``, ROI labels, ``band``, ``band_label``,
        ``absolute_band_power``, ``relative_power``.
    """
    bands = list(bands)
    freqs = np.asarray(freqs, dtype=float)
    psd = psd_df[list(freqs)].to_numpy(dtype=float)
    freq_step = float(np.median(np.diff(freqs)))
    total_power = np.nansum(psd, axis=1) * freq_step

    meta_cols = [
        c for c in ["Region name", "Lobe", "hemisphere", "mni_x"] if c in psd_df
    ]
    metadata = add_roi_hemisphere_labels(psd_df[meta_cols]).reset_index(drop=True)
    metadata.insert(0, "channel", psd_df.index.astype(str))
    metadata["dataset"] = dataset

    out = []
    for i, band in enumerate(bands):
        if i == len(bands) - 1:
            mask = (freqs >= band.lo) & (freqs <= band.hi)
        else:
            mask = (freqs >= band.lo) & (freqs < band.hi)
        band_power = np.nansum(psd[:, mask], axis=1) * freq_step
        df = metadata.copy()
        df["band"] = band.name
        df["band_label"] = BAND_LABELS[band.name]
        df["absolute_band_power"] = band_power
        df["relative_power"] = np.divide(
            band_power,
            total_power,
            out=np.full_like(band_power, np.nan, dtype=float),
            where=total_power > 0,
        )
        out.append(df)
    return pd.concat(out, ignore_index=True)


def region_band_power(
    channel_band_power: pd.DataFrame,
    summary: Summary = "mean",
    roi_level: Literal["hemisphere", "bilateral"] = "hemisphere",
) -> pd.DataFrame:
    """Collapse channels into a normative map at the bilateral or hemisphere level.

    Parameters
    ----------
    channel_band_power : DataFrame
        Output of :func:`relative_band_power_by_channel` (one or both modalities).
    summary : {"mean", "median"} or callable, optional
        Aggregator for collapsing channels within an ROI.
    roi_level : {"hemisphere", "bilateral"}, optional
        ``"bilateral"`` pools left/right channels within each region;
        ``"hemisphere"`` keeps left/right ROIs separate.

    Returns
    -------
    DataFrame
        One row per (dataset, ROI, band) with ``relative_power`` (summarized),
        ``n_channels``, and ``sd_channels``.
    """
    summary_func = summary if isinstance(summary, str) else summary
    if roi_level == "hemisphere":
        group_cols = [
            "dataset",
            "Region name",
            "hemisphere",
            "roi_hemisphere",
            "Lobe",
            "band",
            "band_label",
        ]
    elif roi_level == "bilateral":
        group_cols = [
            "dataset",
            "Region name",
            "roi_bilateral",
            "Lobe",
            "band",
            "band_label",
        ]
    else:
        raise ValueError("roi_level must be 'hemisphere' or 'bilateral'.")

    return (
        channel_band_power.groupby(group_cols, observed=True)
        .agg(
            relative_power=("relative_power", summary_func),
            n_channels=("relative_power", "size"),
            sd_channels=("relative_power", "std"),
        )
        .reset_index()
    )


def compare_region_band_power(
    region_power: pd.DataFrame,
    x_dataset: str,
    y_dataset: str,
    roi_col: str = "roi_hemisphere",
    n_label: str = "n_roi_hemispheres",
    bands: Iterable[Band] = EEG_BANDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair two regional band-power maps and compute Spearman rho per band.

    Parameters
    ----------
    region_power : DataFrame
        Output of :func:`region_band_power`, containing both modalities.
    x_dataset, y_dataset : str
        ``dataset`` values placed on the x- and y-axes respectively.
    roi_col : str, optional
        Column identifying the shared spatial unit to merge on
        ("roi_bilateral" or "roi_hemisphere").
    n_label : str, optional
        Name of the per-band sample-size column in the correlations table.
    bands : iterable of Band, optional
        Defaults to :data:`pesco.experimental.clustering.EEG_BANDS`.

    Returns
    -------
    paired : DataFrame
        One row per (ROI, band) with ``x_relative_power`` and
        ``y_relative_power`` (plus per-side channel counts / SDs).
    correlations : DataFrame
        One row per band with ``spearman_rho``, ``p_value`` and ``n_label``.
        rho/p are NaN for bands with fewer than 3 paired ROIs.
    """
    bands = list(bands)
    x_meta_cols = [roi_col, "Region name", "Lobe", "band", "band_label"]
    if "hemisphere" in region_power.columns:
        x_meta_cols.insert(2, "hemisphere")

    x = (
        region_power.loc[region_power["dataset"] == x_dataset]
        .rename(
            columns={
                "relative_power": "x_relative_power",
                "n_channels": "x_n_channels",
                "sd_channels": "x_sd_channels",
            }
        )[x_meta_cols + ["x_relative_power", "x_n_channels", "x_sd_channels"]]
    )
    y = (
        region_power.loc[region_power["dataset"] == y_dataset]
        .rename(
            columns={
                "relative_power": "y_relative_power",
                "n_channels": "y_n_channels",
                "sd_channels": "y_sd_channels",
            }
        )[[roi_col, "band", "y_relative_power", "y_n_channels", "y_sd_channels"]]
    )
    paired = x.merge(y, on=[roi_col, "band"], how="inner")

    rows = []
    for band in bands:
        d = paired.loc[paired["band"] == band.name].dropna(
            subset=["x_relative_power", "y_relative_power"]
        )
        if len(d) >= 3:
            rho, p_value = stats.spearmanr(
                d["x_relative_power"], d["y_relative_power"]
            )
        else:
            rho, p_value = np.nan, np.nan
        rows.append(
            {
                "band": band.name,
                "band_label": BAND_LABELS[band.name],
                "spearman_rho": rho,
                "p_value": p_value,
                n_label: len(d),
            }
        )
    correlations = pd.DataFrame(rows)
    return paired, correlations


def plot_band_power_correlation_grid(
    paired: pd.DataFrame,
    correlations: pd.DataFrame,
    color_by: ColorBy = "band",
    x_label: str = "source HD-EEG\nrelative band power",
    y_label: str = "intracranial EEG\nrelative band power",
    title: str | None = "Uncorrected PSD: relative band-power correspondence",
    x_col: str = "x_relative_power",
    y_col: str = "y_relative_power",
    bands: Iterable[Band] = EEG_BANDS,
) -> Figure:
    """Janiukstyte Fig. 3A style scatter grid: one panel per band.

    Each panel scatters paired regional relative power (x vs y) with a linear
    fit line, and shows the band's Spearman rho in the title.

    Parameters
    ----------
    paired, correlations : DataFrame
        Outputs of :func:`compare_region_band_power`.
    color_by : {"band", "lobe"}, optional
        ``"band"`` (default, matches the reference figure): all points in a
        panel share the band's color and the rho title is drawn in that color.
        ``"lobe"``: points colored by anatomical lobe with a shared legend and
        black titles/fit lines.
    x_label, y_label, title : str
        Axis labels and figure suptitle. ``title=None`` omits the suptitle.
    x_col, y_col : str
        Columns in ``paired`` to plot.
    bands : iterable of Band, optional
        Defaults to :data:`pesco.experimental.clustering.EEG_BANDS`.

    Returns
    -------
    matplotlib.figure.Figure
        The caller is responsible for saving (see module docstring).
    """
    if color_by not in ("band", "lobe"):
        raise ValueError("color_by must be 'band' or 'lobe'.")

    bands = list(bands)
    fig, axes = plt.subplots(1, len(bands), figsize=(15, 3.3))

    for ax, band in zip(axes, bands):
        d = paired.loc[paired["band"] == band.name].dropna(subset=[x_col, y_col])

        if color_by == "band":
            ax.scatter(
                d[x_col],
                d[y_col],
                s=32,
                alpha=0.72,
                color=BAND_COLORS[band.name],
                edgecolors="none",
            )
            fit_color = BAND_COLORS[band.name]
            title_color = BAND_COLORS[band.name]
        else:
            for lobe in LOBE_ORDER:
                lobe_data = d.loc[d["Lobe"] == lobe]
                if lobe_data.empty:
                    continue
                ax.scatter(
                    lobe_data[x_col],
                    lobe_data[y_col],
                    s=32,
                    alpha=0.72,
                    color=LOBE_COLORS[lobe],
                    edgecolors="none",
                )
            fit_color = "black"
            title_color = "black"

        if len(d) >= 2 and d[x_col].nunique() > 1:
            slope, intercept = np.polyfit(d[x_col], d[y_col], deg=1)
            x_range = np.linspace(d[x_col].min(), d[x_col].max(), 100)
            ax.plot(x_range, slope * x_range + intercept, color=fit_color, lw=2.2)

        rho = correlations.loc[
            correlations["band"] == band.name, "spearman_rho"
        ].iloc[0]
        ax.set_title(
            f"{BAND_LABELS[band.name]}\nrho={rho:.2f}",
            color=title_color,
            fontsize=15,
            pad=8,
        )
        ax.set_xlabel(x_label, fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(y_label, fontsize=11)
    if title is not None:
        fig.suptitle(title, y=1.08, fontsize=14)

    if color_by == "lobe":
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                label=lobe,
                markerfacecolor=LOBE_COLORS[lobe],
                markeredgecolor="none",
                markersize=7,
                alpha=0.72,
            )
            for lobe in LOBE_ORDER
            if lobe in set(paired["Lobe"])
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=10,
        )

    fig.tight_layout()
    return fig
