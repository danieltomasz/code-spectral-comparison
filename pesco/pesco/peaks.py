from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from mizani.formatters import percent_format
from plotnine import (
    aes,
    element_text,
    geom_col,
    geom_density,
    geom_vline,
    ggplot,
    labs,
    position_dodge,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)
from specparam import SpectralGroupModel

from pesco import config
from pesco.config import DATASETS
from pesco.experimental.clustering import Band, EEG_BANDS, band_edges
from pesco.spectral import specparam2pandas

# Marker colours for the two modalities (dumbbell plot).
DATASET_COLORS = {"iEEG atlas": "#2C7FB8", "source HD-EEG": "#E8546B"}


def _subject_id(meta: pd.DataFrame) -> np.ndarray:
    """Subject (cluster) id per channel, matching the BIC model-selection build.

    iEEG atlas channels carry a ``patient`` column -> ``ieeg_p<patient>``. Source
    HD-EEG channel names are ``dataset<NN>_<region><hemi>``, so the subject is the
    prefix before the underscore (the ``dataset`` column is constant 'sources').
    """
    if "patient" in meta.columns:
        return ("ieeg_p" + meta["patient"].astype(str)).to_numpy()
    return meta.index.astype(str).str.split("_").str[0].to_numpy()


def dataset_peaks(
    path: str | Path,
    dataset: str,
    mode: str,
    settings: dict | None = None,
    freq_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Fit one modality's saved PSD CSV and return the tidy peak backbone.

    Loads the normalized-PSD table at ``path``, fits the aperiodic ``mode`` with
    the canonical specparam ``settings`` over ``freq_range`` (both default to
    :mod:`pesco.config`), and returns :func:`pesco.spectral.specparam2pandas`
    output with per-channel metadata attached: ``channel``, ``region``, ``Lobe``,
    ``subject``, ``dataset``.

    No-peak spectra are kept as rows with NaN ``CF``/``PW``/``BW`` so the channel
    set stays the full prevalence denominator; drop on ``CF`` where a per-peak
    view is needed.
    """
    settings = config.SPECPARAM_SETTINGS if settings is None else settings
    freq_range = config.FREQ_RANGE if freq_range is None else freq_range

    df = pd.read_csv(path, index_col=0)
    num = [c for c in df.columns if str(c).replace(".", "", 1).isdigit()]
    freqs = np.asarray(num, dtype=float)
    psd = df[num].to_numpy(dtype=float)
    meta = df.drop(columns=num)

    fg = SpectralGroupModel(**settings, aperiodic_mode=mode, verbose=False)
    fg.fit(freqs, psd, freq_range=list(freq_range), n_jobs=1)

    chan_meta = pd.DataFrame(
        {
            "channel": meta.index.astype(str),
            "region": meta["Region name"].astype(str).str.strip("'"),
            "Lobe": meta["Lobe"].astype(str).str.strip("'"),
            "subject": _subject_id(meta),
        }
    ).reset_index(drop=True)

    peaks = specparam2pandas(fg)
    peaks["channel"] = chan_meta["channel"].to_numpy()[peaks["ID"].astype(int)]
    return peaks.merge(chan_meta, on="channel", how="left").assign(dataset=dataset)


def plot_peak_centre_frequency(
    peaks: pd.DataFrame,
    datasets: Sequence[str] = DATASETS,
    bands: Iterable[Band] = EEG_BANDS,
    breaks: Sequence[float] = (1, 4, 8, 13, 30, 50, 80),
    title: str = "Fitted peak centre frequencies vs canonical band edges",
) -> ggplot:
    """Density of fitted peak centre frequencies per modality, vs canonical band edges.

    One filled density of the ``CF`` column per ``dataset``, with the canonical
    band edges drawn as dashed vlines. Per the project convention the figure is
    returned only; the caller saves it.

    Parameters
    ----------
    peaks : DataFrame
        Tidy peak table (e.g. from :func:`dataset_peaks`), with a ``CF``
        centre-frequency column and a ``dataset`` label column. No-peak rows
        (NaN ``CF``) are dropped here, so the full backbone can be passed in.
    datasets : sequence of str, optional
        Modality labels, in plotting order. Defaults to
        :data:`pesco.config.DATASETS`.
    bands : iterable of Band, optional
        Canonical bands whose edges are drawn as dashed vlines. Defaults to
        :data:`pesco.experimental.clustering.EEG_BANDS`.
    breaks : sequence of float, optional
        x-axis tick positions (Hz).
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    d = peaks.dropna(subset=["CF"]).assign(
        dataset=lambda x: pd.Categorical(x["dataset"], categories=list(datasets), ordered=True)
    )
    return (
        ggplot(d, aes("CF", fill="dataset"))
        + geom_vline(
            xintercept=band_edges(bands), linetype="dashed", color="#999999", size=0.4
        )
        + geom_density(alpha=0.45, color="none")
        + scale_x_continuous(breaks=list(breaks))
        + labs(x="peak centre frequency (Hz)", y="density", fill="", title=title)
        + theme_bw()
        + theme(figure_size=(8, 3.5), plot_title=element_text(size=11))
    )


def _boot_mode(sub: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> dict:
    """Subject-bootstrap modal snapped-CF for one region/modality block.

    ``sub`` is the region's full channel set (a ``cf_bin`` of the highest-peak
    centre frequency where a peak was fitted, NaN otherwise, plus ``subject`` and
    ``has_peak``). Resamples the peak-bearing subjects with replacement and, per
    draw, takes the modal ``cf_bin``; returns the mode of the per-draw modes with
    a 2.5/97.5 percentile interval, the peak-detection rate (no-peak channels are
    the miss count), and the subject count. Frequencies are NaN where the region
    carries no peak at all.
    """
    n_total = len(sub)
    peaked = sub[sub["has_peak"]]
    det = len(peaked) / n_total if n_total else np.nan
    n_subj = sub["subject"].nunique()
    subj_ids = peaked["subject"].dropna().unique()
    if len(subj_ids) == 0:
        return {"mode": np.nan, "lo": np.nan, "hi": np.nan, "detection": det, "n_subjects": n_subj}
    by_subj = {s: peaked.loc[peaked["subject"] == s, "cf_bin"].to_numpy() for s in subj_ids}
    k = len(subj_ids)
    modes = np.empty(n_boot)
    for b in range(n_boot):
        draw = np.concatenate([by_subj[s] for s in rng.choice(subj_ids, size=k, replace=True)])
        vals, counts = np.unique(draw, return_counts=True)
        modes[b] = vals[counts.argmax()]  # ties -> lowest frequency
    v, c = np.unique(modes, return_counts=True)
    lo, hi = np.percentile(modes, [2.5, 97.5])
    return {"mode": float(v[c.argmax()]), "lo": float(lo), "hi": float(hi), "detection": det, "n_subjects": n_subj}


def region_dominant_frequency_bootmode(
    peaks: pd.DataFrame,
    datasets: Sequence[str] = DATASETS,
    grid: float = 0.5,
    n_boot: int = 2000,
    seed: int = 3,
) -> pd.DataFrame:
    """Per-region dominant peak frequency by subject-bootstrap mode.

    For each channel the dominant peak is its highest-amplitude (max ``PW``) peak;
    its centre frequency is snapped to a ``grid`` Hz lattice. Within each region x
    modality the channels' subjects are resampled with replacement (``n_boot``
    draws) and the modal snapped frequency is taken per draw — robust to the
    single loud channel that the plain argmax would track. Channels with no fitted
    peak have no dominant frequency and are excluded from the mode, but are kept as
    the miss count behind the ``detection`` rate (see :func:`dataset_peaks` for the
    NaN-CF backbone rows).

    Returns one row per (dataset, region): ``Lobe``, ``mode`` (point), ``lo`` /
    ``hi`` (2.5/97.5 bootstrap percentile), ``detection`` (peak-detection rate),
    ``n_subjects``.
    """
    pk = peaks.dropna(subset=["CF", "PW"])
    idx = pk.groupby(["dataset", "channel"], observed=True)["PW"].idxmax()
    dom = pk.loc[idx, ["dataset", "channel", "CF"]].copy()
    dom["cf_bin"] = np.round(dom["CF"].to_numpy() / grid) * grid
    chan = (
        peaks[["dataset", "channel", "region", "Lobe", "subject"]]
        .drop_duplicates()
        .merge(dom[["dataset", "channel", "cf_bin"]], on=["dataset", "channel"], how="left")
    )
    chan["has_peak"] = chan["cf_bin"].notna()

    rng = np.random.default_rng(seed)
    recs = [
        {"dataset": ds, "region": region, "Lobe": sub["Lobe"].iloc[0], **_boot_mode(sub, rng, n_boot)}
        for (ds, region), sub in chan.groupby(["dataset", "region"], observed=True)
    ]
    out = pd.DataFrame(recs)
    out["dataset"] = pd.Categorical(out["dataset"], list(datasets), ordered=True)
    return out


def plot_dominant_frequency_dumbbell(
    dom_freq: pd.DataFrame,
    datasets: Sequence[str] = DATASETS,
    dataset_colors: dict[str, str] | None = None,
    band_breaks: Sequence[float] = (1, 4, 8, 13, 30, 80),
    figsize: tuple[float, float] = (8, 11),
    title: str = "Dominant peak frequency per region (subject-bootstrap mode of highest-peak CF)",
):
    """Dumbbell of regional dominant peak frequency, iEEG vs source HD-EEG.

    Takes :func:`region_dominant_frequency_bootmode` output. Regions are stacked
    in lobe blocks (matching the chapter's heatmaps, first lobe on top) with
    lobe-coloured labels; each region's two modality points (``mode``) are joined
    by a grey segment, on a log frequency x-axis with canonical band edges drawn.
    Per the project convention the figure is returned only; the caller saves it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    from pesco.experimental.plotting import _LOBE_COLORS, _LOBE_ORDER

    dataset_colors = DATASET_COLORS if dataset_colors is None else dataset_colors
    rank = {lo: i for i, lo in enumerate(_LOBE_ORDER)}
    region_lobe = dom_freq.drop_duplicates("region").set_index("region")["Lobe"].to_dict()
    # lobe blocks then region name; reversed so the first lobe block sits on top
    order = list(reversed(sorted(region_lobe, key=lambda r: (rank.get(region_lobe[r], len(_LOBE_ORDER)), r))))
    ypos = {r: i for i, r in enumerate(order)}
    wide = dom_freq.pivot_table(index="region", columns="dataset", values="mode", observed=True)

    fig, ax = plt.subplots(figsize=figsize)
    for e in band_breaks:
        ax.axvline(e, ls="--", color="#cccccc", lw=0.4)
    for r in order:  # connecting segment where both modalities have a mode
        if r in wide.index and wide.loc[r, list(datasets)].notna().all():
            ax.plot(
                [wide.loc[r, datasets[0]], wide.loc[r, datasets[1]]],
                [ypos[r], ypos[r]], color="#bdbdbd", lw=1.5, zorder=1,
            )
    for ds in datasets:
        sub = dom_freq[dom_freq["dataset"] == ds].dropna(subset=["mode"])
        ax.scatter(sub["mode"], sub["region"].map(ypos), s=44, color=dataset_colors[ds], edgecolor="none", zorder=3, label=ds)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    for t, r in zip(ax.get_yticklabels(), order):
        t.set_color(_LOBE_COLORS.get(region_lobe.get(r), "black"))
    ax.set_xscale("log")
    ax.set_xticks(list(band_breaks))
    ax.xaxis.set_major_formatter(ScalarFormatter())  # plain numbers, not 10^x
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.95, max(band_breaks) + 5)
    ax.set_ylim(-1, len(order))
    ax.set_xlabel("dominant peak frequency (Hz, log scale)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.tick_params(length=0)
    return fig


def frauscher_bands(freq_range: tuple[float, float] | None = None) -> list[Band]:
    """Frauscher 2018's 22 data-driven frequency intervals, as Band objects.

    The paper's interval edges over ``freq_range`` (default
    :data:`pesco.config.FREQ_RANGE`); same construction as the overlap figures in
    ``1b_overlap_in_bands``. Each band is named ``"<lo>-<hi>"`` for the heatmap
    x-axis.
    """
    from pesco.experimental.peak_testing import cutintervals

    lo, hi = config.FREQ_RANGE if freq_range is None else freq_range
    _, edges = cutintervals(np.asarray([lo, hi], dtype=float))
    return [
        Band(float(a), float(b), f"{a:g}-{b:g}", "", float((a + b) / 2))
        for a, b in zip(edges[:-1], edges[1:])
    ]


def _assign_band(cf: float, bands: Sequence[Band]) -> str | None:
    """Name of the band whose half-open [lo, hi) contains ``cf`` (last band closed)."""
    for i, b in enumerate(bands):
        if (b.lo <= cf < b.hi) or (i == len(bands) - 1 and cf == b.hi):
            return b.name
    return None


def region_band_prevalence(
    peaks: pd.DataFrame,
    bands: Sequence[Band],
    datasets: Sequence[str] = DATASETS,
) -> dict[str, pd.DataFrame]:
    """region x band peak prevalence per modality, for any band scheme.

    Prevalence is the fraction of a region's channels with at least one fitted
    peak whose centre frequency falls in the band; the denominator is the region's
    full channel set, so no-peak channels (the NaN-CF backbone rows from
    :func:`dataset_peaks`) count as 0. Returns one region x band frame per dataset.
    """
    names = [b.name for b in bands]
    pk = peaks.dropna(subset=["CF"]).copy()
    pk["sb"] = pk["CF"].map(lambda cf: _assign_band(cf, bands))
    has = pk.dropna(subset=["sb"])[["dataset", "channel", "sb"]].drop_duplicates()
    has["has_peak"] = True
    chans = peaks[["dataset", "channel", "region"]].drop_duplicates()
    grid = (
        chans.assign(key=1)
        .merge(pd.DataFrame({"sb": names, "key": 1}), on="key")
        .drop(columns="key")
        .merge(has, on=["dataset", "channel", "sb"], how="left")
    )
    grid["has_peak"] = grid["has_peak"].notna()
    prev = (
        grid.groupby(["dataset", "region", "sb"], observed=True)["has_peak"]
        .mean()
        .reset_index(name="prevalence")
    )
    return {
        ds: prev[prev["dataset"] == ds]
        .pivot(index="region", columns="sb", values="prevalence")
        .reindex(columns=names)
        for ds in datasets
    }


def plot_peak_prevalence_grid(
    peaks: pd.DataFrame,
    canonical: Sequence[Band] = EEG_BANDS,
    datasets: Sequence[str] = DATASETS,
    cmap: str = "rocket_r",
):
    """4-panel region x band peak-prevalence heatmap: canonical bands | Frauscher intervals.

    iEEG (top, A-B) over source HD-EEG (bottom, C-D), each modality in the five
    canonical bands and the 22 Frauscher intervals, on a shared 0-1 colour scale
    (the prevalence twin of the aperiodic-removed power heatmap in
    ``1b_overlap_in_bands``). Built on the peak backbone from
    :func:`dataset_peaks` and reuses
    :func:`pesco.experimental.facets.overlap_heatmap_grid`. Returns the matplotlib
    Figure; the caller saves.
    """
    from pesco.bandpower import BAND_LABELS
    from pesco.experimental.facets import overlap_heatmap_grid

    canon = region_band_prevalence(peaks, list(canonical), datasets)
    canon = {ds: df.rename(columns=BAND_LABELS) for ds, df in canon.items()}  # symbols -> δ θ α β γ words
    frausch = region_band_prevalence(peaks, frauscher_bands(), datasets)

    region_lobe = (
        peaks.dropna(subset=["region"])
        .drop_duplicates("region")
        .set_index("region")["Lobe"]
        .to_dict()
    )
    a, b = datasets[0], datasets[1]
    regions = sorted(set(canon[a].index) & set(canon[b].index))
    panels = [
        {"df": canon[a].reindex(regions), "title": f"{a} · canonical", "xlabel": "Frequency band"},
        {"df": frausch[a].reindex(regions), "title": f"{a} · Frauscher", "xlabel": "Frauscher interval (Hz)"},
        {"df": canon[b].reindex(regions), "title": f"{b} · canonical", "xlabel": "Frequency band"},
        {"df": frausch[b].reindex(regions), "title": f"{b} · Frauscher", "xlabel": "Frauscher interval (Hz)"},
    ]
    fig, _ = overlap_heatmap_grid(
        panels,
        region_lobe,
        vmin=0.0,
        vmax=1.0,  # prevalence is bounded [0, 1]; one shared scale across panels
        nrows=2,  # iEEG (top) over HD-EEG (bottom); canonical | Frauscher across cols
        cmap=cmap,
        cbar_label="peak prevalence",
        row_height=0.32,
        tick_fontsize=13,
        ytick_fontsize=12,
        title_fontsize=15,
        axis_label_fontsize=12,
    )
    return fig


def no_peak_fraction(
    peaks: pd.DataFrame,
    group: str = "Lobe",
    datasets: Sequence[str] = DATASETS,
) -> pd.DataFrame:
    """Per-(dataset, group) fraction of channels with no fitted peak.

    A channel counts as no-peak when *all* its rows have a NaN ``CF``, so the
    backbone must keep the no-peak rows (see :func:`dataset_peaks`). ``group`` is
    any channel-level column, e.g. ``"Lobe"`` or ``"region"``.

    Returns one row per (dataset, group) with ``n_channels``, ``n_no_peak`` and
    ``fraction``.
    """
    chan = (
        peaks.groupby(["dataset", "channel", group], observed=True)["CF"]
        .agg(lambda s: bool(s.isna().all()))
        .reset_index(name="no_peak")
    )
    out = (
        chan.groupby(["dataset", group], observed=True)
        .agg(n_channels=("channel", "size"), n_no_peak=("no_peak", "sum"))
        .reset_index()
    )
    out["fraction"] = out["n_no_peak"] / out["n_channels"]
    out["dataset"] = pd.Categorical(out["dataset"], list(datasets), ordered=True)
    return out


def no_peak_table(
    peaks: pd.DataFrame,
    group: str = "Lobe",
    datasets: Sequence[str] = DATASETS,
    group_order: Sequence[str] | None = None,
    overall_label: str = "All lobes",
) -> pd.DataFrame:
    """Publication-ready wide table of no-peak channel counts per ``group`` and modality.

    One row per ``group`` value (in ``group_order``, else sorted) plus an
    ``overall_label`` total row; two columns per dataset — the channel count and
    the no-peak count formatted ``n (pct%)``. Render with ``df.to_markdown(
    index=False)`` (see :func:`no_peak_fraction` for the raw numbers).
    """
    frac = no_peak_fraction(peaks, group=group, datasets=datasets)
    chan = (
        peaks.groupby(["dataset", "channel"], observed=True)["CF"]
        .agg(lambda s: bool(s.isna().all()))
        .reset_index(name="no_peak")
    )
    overall = (
        chan.groupby("dataset", observed=True)
        .agg(n_channels=("channel", "size"), n_no_peak=("no_peak", "sum"))
        .reset_index()
        .assign(**{group: overall_label})
    )
    overall["fraction"] = overall["n_no_peak"] / overall["n_channels"]
    long = pd.concat([frac, overall], ignore_index=True)
    long["cell"] = long.apply(
        lambda r: f"{int(r.n_no_peak)} ({r.fraction * 100:.1f}%)", axis=1
    )

    order = list(group_order) if group_order is not None else sorted(frac[group].unique())
    order = [g for g in order if g in set(frac[group])] + [overall_label]
    out = pd.DataFrame({group: order})
    for ds in datasets:
        sub = long[long["dataset"] == ds].set_index(group)
        out[f"{ds}: channels"] = out[group].map(sub["n_channels"]).astype(int)
        out[f"{ds}: no peak"] = out[group].map(sub["cell"])
    return out


def plot_no_peak_fraction(
    fraction: pd.DataFrame,
    group: str = "Lobe",
    group_order: Sequence[str] | None = None,
    datasets: Sequence[str] = DATASETS,
    colors: dict[str, str] | None = None,
    title: str = "Channels with no fitted peak",
) -> ggplot:
    """Grouped bar of the no-peak channel fraction per ``group`` and modality.

    Takes the output of :func:`no_peak_fraction`. Per the project convention the
    figure is returned only; the caller saves it.
    """
    d = fraction.copy()
    if group_order is not None:
        d[group] = pd.Categorical(d[group], categories=list(group_order), ordered=True)
    d["dataset"] = pd.Categorical(d["dataset"], categories=list(datasets), ordered=True)
    p = (
        ggplot(d, aes(group, "fraction", fill="dataset"))
        + geom_col(position=position_dodge(width=0.8), width=0.7)
        + scale_y_continuous(labels=percent_format())
        + labs(x="", y="channels with no fitted peak", title=title)
        + theme_bw()
        + theme(figure_size=(7, 3.8), plot_title=element_text(size=11))
    )
    return p + scale_fill_manual(values=colors, name="") if colors else p


def peak_survival(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    region_col: str = "region",
    band_col: str = "band",
    present_col: str = "peak_present",
) -> pd.DataFrame:
    """Compare regional peak prevalence before and after aperiodic correction.

    Takes two DataFrames — each with one row per region x band combination and a
    boolean presence column — and returns a merged table showing which peaks
    survived correction, were lost, or appeared for the first time.

    Parameters
    ----------
    before_df : pandas.DataFrame
        Peak prevalence before correction.  Must contain ``region_col``,
        ``band_col``, and ``present_col`` columns.
    after_df : pandas.DataFrame
        Peak prevalence after correction.  Same schema as ``before_df``.
    region_col : str, optional, default: "region"
        Column name identifying brain region.
    band_col : str, optional, default: "band"
        Column name identifying frequency band / interval.
    present_col : str, optional, default: "peak_present"
        Boolean (or 0/1) column indicating whether a significant peak was found.

    Returns
    -------
    pandas.DataFrame
        One row per region × band, with columns:
        ``region``, ``band``, ``before``, ``after``,
        ``survived`` (before & after),
        ``lost``     (before & ~after),
        ``gained``   (not before & after).
    """
    for label, df in [("before_df", before_df), ("after_df", after_df)]:
        missing = {region_col, band_col, present_col} - set(df.columns)
        if missing:
            raise ValueError(
                f"{label} is missing columns: {missing}. "
                f"Expected '{region_col}', '{band_col}', '{present_col}'."
            )

    key_cols = [region_col, band_col]

    merged = before_df[key_cols + [present_col]].merge(
        after_df[key_cols + [present_col]],
        on=key_cols,
        how="outer",
        suffixes=("_before", "_after"),
    ).fillna(False)

    before = merged[f"{present_col}_before"].astype(bool)
    after  = merged[f"{present_col}_after"].astype(bool)

    result = merged[key_cols].copy()
    result.columns = [region_col, band_col]
    result["before"]   = before
    result["after"]    = after
    result["survived"] = before & after
    result["lost"]     = before & ~after
    result["gained"]   = ~before & after

    return result.reset_index(drop=True)
