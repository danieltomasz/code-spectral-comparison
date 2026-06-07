"""Generic faceting for the bespoke matplotlib panel renderers.

Separates three concerns that the hand-written ``*_pair`` functions tangle:

* a **panel renderer** draws one cell into a given ``ax`` (e.g. :func:`psd_panel`),
* :class:`FacetGrid` lays panels out across two facet dimensions and owns the
  shared axes, header strips, single shared labels, suptitle, spacing and
  "journal" styling,
* the style layer is a plain rcParams dict applied once.

A figure like "PSD per lobe (rows) × modality (columns)" becomes::

    FacetGrid(row=GroupBy("Lobe"), col=Bundles([ieeg, src], label="name"))
        .map(psd_panel(summary="median"))
        .label(x="Frequency [Hz]", y="Normalized spectral density")
        .finish(suptitle="Lobar differences: iEEG vs reconstructed sources")

Swap ``row``/``col`` to transpose. The grid does not save; the caller saves
the returned figure.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from pesco.experimental.plotting import (
    _LOBE_ORDER,
    _build_region_rows_meta,
    _plot_subplot,
    _prepare_region_df,
    plot_clusters,
    plot_overlap_frauscher_heatmap,
    plot_region_difference_heatmap,
)

_CANONICAL_LOBES = ("Occipital", "Parietal", "Frontal", "Temporal")

JOURNAL_STYLE: dict = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.labelsize": 12,   # supxlabel / supylabel (default "large" is oversized)
    "figure.titlesize": 13,   # suptitle
    # NB: do not set axes.grid here — the renderers call ax.grid() (a toggle),
    # so an rc default of True would flip the grid back off.
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
_JOURNAL_STYLE = JOURNAL_STYLE  # backwards-compatible private alias


# ---------------------------------------------------------------------------
# facet dimensions
# ---------------------------------------------------------------------------

@dataclass
class Bundles:
    """A facet axis bound to a list of dataset bundles (e.g. ``DatasetCtx``).

    Each item becomes one facet level; its header label is ``getattr(item,
    label)`` (default the item's ``name``). The item itself is handed to the
    panel renderer as ``cell.bundle``.
    """

    items: list
    label: str = "name"

    def label_of(self, item) -> str:
        return str(getattr(item, self.label, item))


@dataclass
class GroupBy:
    """A facet axis bound to a categorical column of each bundle's ``psd``.

    Levels are the values present in ``column`` (unioned across bundles).
    ``order`` pins an explicit order; otherwise canonical lobe order is used
    for ``"Lobe"`` and first-seen order elsewhere.
    """

    column: str
    order: list | None = None

    def categories(self, bundles: list) -> list:
        seen: list = []
        for b in bundles:
            for v in b.psd[self.column].dropna().unique():
                if v not in seen:
                    seen.append(v)
        ref = list(self.order) if self.order else (
            list(_CANONICAL_LOBES) if self.column == "Lobe" else []
        )
        return [v for v in ref if v in seen] + [v for v in seen if v not in ref]


Dim = Bundles | GroupBy | None


@dataclass
class FacetCell:
    """One panel: its axes, the bundle it draws from, and the facet coords."""

    ax: Axes
    bundle: object
    coords: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# grid manager
# ---------------------------------------------------------------------------

class FacetGrid:
    """Lay bespoke panel renderers out across two facet dimensions.

    Exactly one of ``row``/``col`` must be a :class:`Bundles` (so each cell
    knows which bundle to draw); the other may be a :class:`GroupBy` or
    ``None`` (single level). Drawing is deferred to :meth:`finish` so a single
    rcParams context covers every artist.
    """

    def __init__(
        self,
        *,
        row: Dim = None,
        col: Dim = None,
        sharex: bool = True,
        sharey: bool = True,
        panel_size: tuple[float, float] = (5.0, 3.0),
        style: str | dict | None = "journal",
        wspace: float = 0.08,
        hspace: float = 0.32,
    ) -> None:
        if sum(isinstance(d, Bundles) for d in (row, col)) != 1:
            raise ValueError("exactly one of row/col must be a Bundles dimension")
        self.row = row
        self.col = col
        self.sharex = sharex
        self.sharey = sharey
        self.panel_size = panel_size
        self.style = (
            _JOURNAL_STYLE if style == "journal"
            else {} if style is None else dict(style)
        )
        self.wspace = wspace
        self.hspace = hspace
        self._panel_fn: Callable[[FacetCell], None] | None = None
        self._xlabel: str | None = None
        self._ylabel: str | None = None

    # -- fluent configuration ------------------------------------------------
    def map(self, panel_fn: Callable[[FacetCell], None]) -> "FacetGrid":
        self._panel_fn = panel_fn
        return self

    def label(self, *, x: str | None = None, y: str | None = None) -> "FacetGrid":
        self._xlabel, self._ylabel = x, y
        return self

    # -- internals -----------------------------------------------------------
    @property
    def _bundles(self) -> list:
        dim = self.row if isinstance(self.row, Bundles) else self.col
        return dim.items

    def _levels(self, dim: Dim) -> list[tuple]:
        if dim is None:
            return [(None, "")]
        if isinstance(dim, Bundles):
            return [(it, dim.label_of(it)) for it in dim.items]
        return [(c, str(c)) for c in dim.categories(self._bundles)]

    def _cell(self, ax: Axes, row_val, col_val) -> FacetCell:
        bundle = row_val if isinstance(self.row, Bundles) else col_val
        coords: dict = {}
        if isinstance(self.row, GroupBy):
            coords[self.row.column] = row_val
        if isinstance(self.col, GroupBy):
            coords[self.col.column] = col_val
        return FacetCell(ax=ax, bundle=bundle, coords=coords)

    # -- draw ----------------------------------------------------------------
    def finish(self, suptitle: str | None = None) -> Figure:
        if self._panel_fn is None:
            raise ValueError("call .map(panel_fn) before .finish()")
        row_levels = self._levels(self.row)
        col_levels = self._levels(self.col)
        nrow, ncol = len(row_levels), len(col_levels)

        with plt.rc_context(self.style):
            fig, axes = plt.subplots(
                nrow, ncol,
                figsize=(ncol * self.panel_size[0], nrow * self.panel_size[1]),
                squeeze=False,
            )
            for ri, (row_val, _) in enumerate(row_levels):
                for ci, (col_val, _) in enumerate(col_levels):
                    self._panel_fn(self._cell(axes[ri, ci], row_val, col_val))

            self._share_and_strip(axes, row_levels, col_levels)
            # set margins first so axes positions are final before placing labels
            fig.subplots_adjust(wspace=self.wspace, hspace=self.hspace)
            if self._xlabel:
                fig.supxlabel(self._xlabel)
            if self._ylabel:
                # hug the axes: place the label just left of the left column's
                # tick labels rather than at the figure edge (default x=0.02)
                x0 = min(ax.get_position().x0 for ax in axes[:, 0])
                fig.supylabel(self._ylabel, x=max(x0 - 0.045, 0.005))
            if suptitle:
                fig.suptitle(suptitle)
        return fig

    def _share_and_strip(self, axes, row_levels, col_levels) -> None:
        nrow, ncol = axes.shape
        if self.sharey:
            # union of limits — works for linear panels (bottom 0) and log
            lo = min(ax.get_ylim()[0] for ax in axes.flat)
            hi = max(ax.get_ylim()[1] for ax in axes.flat)
            # snap the linear top up to a nice round value (clean ticks, no
            # clipping); log panels keep matplotlib's native decade ticks
            if not any(ax.get_yscale() == "log" for ax in axes.flat):
                ticks = MaxNLocator(steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
                hi = next((t for t in ticks if t >= hi), hi)
            for ax in axes.flat:
                ax.set_ylim(lo, hi)
        for ri in range(nrow):
            for ci in range(ncol):
                ax = axes[ri, ci]
                ax.set_xlabel("")
                ax.set_ylabel("")
                if self.sharey and ci > 0:
                    ax.tick_params(labelleft=False)
                if self.sharex and ri < nrow - 1:
                    ax.tick_params(labelbottom=False)
        # column headers across the top, row headers down the right —
        # plain weight (journals reserve bold for panel letters, not strips)
        for ci, (_, clab) in enumerate(col_levels):
            if clab:
                axes[0, ci].set_title(clab)
        for ri, (_, rlab) in enumerate(row_levels):
            if rlab:
                axes[ri, ncol - 1].annotate(
                    rlab, xy=(1.02, 0.5), xycoords="axes fraction",
                    rotation=-90, ha="left", va="center",
                )


# ---------------------------------------------------------------------------
# panel adapters
# ---------------------------------------------------------------------------

def psd_panel(
    *,
    summary: str = "mean",
    xlim: tuple[float, float] = (1.0, 80.0),
    tick_labelsize: float = 10.0,
    sig_attr: str = "sig_lobes",
    sig_linewidth: float = 3.5,
    show_count: bool = True,
) -> Callable[[FacetCell], None]:
    """Adapter: per-lobe/region PSD panel (IQR + envelope + significance).

    Reads from the cell's bundle (a ``DatasetCtx``): ``psd`` filtered by the
    cell coords, ``no_peak_center`` as the baseline, and ``sig_attr`` (a
    ``{group: intervals}`` dict) keyed by the cell's group value. Wraps the
    existing :func:`pesco.experimental.plotting._plot_subplot`.
    """

    def draw(cell: FacetCell) -> None:
        ctx = cell.bundle
        cols = [c for c in list(ctx.f) if c in ctx.psd.columns]
        mask = pd.Series(True, index=ctx.psd.index)
        for column, value in cell.coords.items():
            mask &= ctx.psd[column] == value
        subset = ctx.psd.loc[mask, cols]
        center = ctx.no_peak_center if len(ctx.no_peak_center) else None
        key = next(iter(cell.coords.values()), None)
        sig = getattr(ctx, sig_attr, {}).get(key) if key is not None else None
        _plot_subplot(
            subset, center, np.asarray(cols, dtype=float), sig, "",
            ax=cell.ax, summary=summary, tick_labelsize=tick_labelsize, xlim=xlim,
            sig_linewidth=sig_linewidth,
        )
        if show_count:
            cell.ax.annotate(
                f"n={len(subset)}", xy=(0.97, 0.93), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color="0.4",
            )

    return draw


def clusters_panel(
    *,
    summary: str = "mean",
    log_y: bool = True,
    order_by_peak: bool = True,
    peak_freq_range: tuple[float, float] = (1.0, 80.0),
    label_by_band: bool = True,
    legend_fontsize: float = 9.0,
    xlim: tuple[float, float] = (1.0, 80.0),
    title: str = "",
) -> Callable[[FacetCell], None]:
    """Adapter: one overlaid per-cluster PSD panel (wraps :func:`plot_clusters`).

    Faceted over a :class:`Bundles` dimension (modality); each panel reads
    ``psd_clust`` / ``smal`` / ``no_peak_center`` from the cell's bundle.
    """

    def draw(cell: FacetCell) -> None:
        ctx = cell.bundle
        plot_clusters(
            ctx.psd_clust, ctx.f, ctx.name, ctx.smal,
            nopeak=ctx.smal[0] if ctx.smal else None,
            summary=summary, log_y=log_y, order_by_peak=order_by_peak,
            peak_baseline=ctx.no_peak_center, peak_freq_range=peak_freq_range,
            feature_cols=ctx.f, label_by_band=label_by_band,
            ax=cell.ax, show=False, save=False,
            title=title, legend_fontsize=legend_fontsize, xlim=xlim,
        )

    return draw


_RAW_PALETTE = ("#E64B35", "#1FA6B8")  # red (iEEG), teal (HD)


def raw_spectra_panel(
    ax: Axes,
    summary: pd.DataFrame,
    *,
    group_col: str = "dataset",
    colors: dict | None = None,
    xlim: tuple[float, float] = (1.0, 80.0),
    xticks=(1, 4, 8, 13, 30, 80),
    ylabel: str = "Raw spectral density",
    legend: bool = True,
    legend_fontsize: float = 8.0,
) -> Axes:
    """Draw average log-log spectra per group into ``ax`` (matplotlib).

    ``summary`` is the long-form-per-group table with columns ``group_col``,
    ``frequency``, ``median``, ``q25``, ``q75``, ``minimum``, ``maximum``.
    Per group: median (solid), IQR band (fill), min/max (dotted). Used as a
    matplotlib panel so it can share a figure/axes-grid with the cluster
    panels (aligned axes), unlike the standalone plotnine version.
    """
    groups = list(dict.fromkeys(summary[group_col]))
    if colors is None:
        colors = {g: _RAW_PALETTE[i % len(_RAW_PALETTE)] for i, g in enumerate(groups)}
    for g in groups:
        sub = summary[summary[group_col] == g].sort_values("frequency")
        c = colors[g]
        ax.fill_between(sub["frequency"], sub["q25"], sub["q75"], color=c, alpha=0.25, linewidth=0)
        ax.plot(sub["frequency"], sub["median"], color=c, linewidth=1.6)
        ax.plot(sub["frequency"], sub["minimum"], color=c, linewidth=0.7, linestyle=":")
        ax.plot(sub["frequency"], sub["maximum"], color=c, linewidth=0.7, linestyle=":")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(list(xticks))
    ax.set_xlim(*xlim)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.grid(True)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)

    if legend:
        mod = [Line2D([], [], color=colors[g], lw=1.6, label=g) for g in groups]
        stat = [
            Line2D([], [], color="0.2", lw=1.6, ls="-", label="median"),
            Line2D([], [], color="0.2", lw=0.8, ls=":", label="min–max"),
            Patch(facecolor="0.5", alpha=0.25, label="IQR (25–75%)"),
        ]
        leg = ax.legend(handles=mod, loc="lower left", fontsize=legend_fontsize,
                        framealpha=0.9, title="modality")
        ax.add_artist(leg)
        ax.legend(handles=stat, loc="upper right", fontsize=legend_fontsize, framealpha=0.9)
    return ax


# ---------------------------------------------------------------------------
# composing heterogeneous figures (matplotlib + plotnine) into labelled panels
# ---------------------------------------------------------------------------

def _render_to_image(obj, dpi: float) -> np.ndarray:
    """Render a matplotlib ``Figure`` or a plotnine ``ggplot`` to an RGBA array."""
    fig = obj if isinstance(obj, Figure) else obj.draw(show=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return mpimg.imread(buf)


def compose_figures(
    panels: list,
    *,
    ncols: int = 1,
    panel_dpi: float = 200.0,
    label_fontsize: float = 16.0,
    hspace: float = 0.02,
    wspace: float = 0.02,
) -> Figure:
    """Tile already-built figures into one labelled multi-panel figure.

    Backend-agnostic alternative to patchworklib for the case patchworklib
    can't handle: composing a plotnine ``ggplot`` with a *multi-axes*
    matplotlib figure (e.g. a :class:`FacetGrid` result). Each panel is
    rasterised at ``panel_dpi`` and placed on a grid; ``a)``, ``b)`` … panel
    letters are drawn top-left. Returns the composite figure; the caller saves.

    ``panels`` is a list of either the figure objects, or ``(figure, label)``
    tuples (``label=None`` suppresses that panel's letter).
    """
    items = [p if isinstance(p, tuple) else (p, "auto") for p in panels]
    letters = "abcdefghijklmnop"
    imgs, labels = [], []
    for i, (obj, lab) in enumerate(items):
        imgs.append(_render_to_image(obj, panel_dpi))
        labels.append(letters[i] if lab == "auto" else lab)

    n = len(imgs)
    nrows = -(-n // ncols)  # ceil
    heights = [im.shape[0] for im in imgs]
    widths = [im.shape[1] for im in imgs]
    row_h = [max(heights[r * ncols:(r + 1) * ncols] or [1]) for r in range(nrows)]
    col_w = [
        max((widths[c::ncols] or [1])) for c in range(ncols)
    ]
    fig = plt.figure(figsize=(sum(col_w) / panel_dpi, sum(row_h) / panel_dpi))
    gs = fig.add_gridspec(
        nrows, ncols, height_ratios=row_h, width_ratios=col_w,
        hspace=hspace, wspace=wspace,
    )
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        ax.imshow(im)
        ax.axis("off")
        if lab:
            ax.annotate(
                f"{lab})", xy=(0, 1), xycoords="axes fraction",
                xytext=(4, -4), textcoords="offset points",
                ha="left", va="top", fontsize=label_fontsize, fontweight="bold",
            )
    return fig


def compose_panels(
    mosaic,
    renderers: dict,
    *,
    style: dict | None = None,
    figsize: tuple[float, float] = (12.0, 5.0),
    sharey_groups: list[list[str]] | None = None,
    sharex_groups: list[list[str]] | None = None,
    letters: list[tuple[str, str]] | None = None,
    titles: list[tuple[str, list[str]]] | None = None,
    title_fontsize: float = 12.0,
    title_pad: float = 0.035,
    letter_fontsize: float = 15.0,
    letter_format: str = "{})",
    letter_offset: tuple[float, float] = (-34, 8),
    letter_offset_shared: tuple[float, float] = (-4, 8),
    **mosaic_kw,
):
    """Lay out real plots on one shared gridspec so their axes align (vector).

    Generalises the bespoke "raw + clusters" composite: define a
    :meth:`~matplotlib.figure.Figure.subplot_mosaic` layout, then hand each
    named cell a ``renderer(ax)`` callable that draws an existing plot into
    that axes (e.g. ``lambda ax: plot_clusters(..., ax=ax)``). Because every
    panel is a live axes in one figure, their plot areas share the same
    extent — what raster tiling / patchworks-of-images cannot guarantee.

    Parameters
    ----------
    mosaic : nested list / string accepted by ``subplot_mosaic`` (cell names).
    renderers : ``{cell_name: callable(ax)}`` — draws into that cell's axes.
    sharey_groups / sharex_groups : lists of cell-name groups that share an
        axis; the first cell keeps its tick labels, the rest are hidden.
    letters : ``[(letter, cell)]`` — bold ``letter)`` at that cell's top-left.
    titles : ``[(text, [cells])]`` — a title centred over those cells' span.
    ``**mosaic_kw`` is forwarded to ``subplot_mosaic`` (``width_ratios``,
    ``height_ratios``, ``gridspec_kw``, ``empty_sentinel`` …).

    Returns ``(fig, axd)`` where ``axd`` maps cell name → axes; the caller saves.
    """
    with plt.rc_context(JOURNAL_STYLE if style is None else style):
        fig = plt.figure(figsize=figsize)
        axd = fig.subplot_mosaic(mosaic, **mosaic_kw)

        for name, draw in renderers.items():
            draw(axd[name])

        for grp in sharey_groups or []:
            base = axd[grp[0]]
            for nm in grp[1:]:
                axd[nm].sharey(base)
                axd[nm].tick_params(labelleft=False)
        for grp in sharex_groups or []:
            base = axd[grp[0]]
            for nm in grp[1:]:
                axd[nm].sharex(base)
                axd[nm].tick_params(labelbottom=False)

        # cells that share another's y-axis have no left tick labels, so their
        # letter needs a small offset (the default clears y-ticks + y-label)
        shared_secondary = {nm for grp in sharey_groups or [] for nm in grp[1:]}
        for letter, cell in letters or []:
            offset = letter_offset_shared if cell in shared_secondary else letter_offset
            axd[cell].annotate(
                letter_format.format(letter), xy=(0, 1), xycoords="axes fraction",
                xytext=offset, textcoords="offset points",
                fontsize=letter_fontsize, fontweight="bold", va="bottom",
            )
        for text, cells in titles or []:
            boxes = [axd[c].get_position() for c in cells]
            xc = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
            yc = max(b.y1 for b in boxes) + title_pad
            fig.text(xc, yc, text, ha="center", va="bottom", fontsize=title_fontsize)
    return fig, axd


def region_difference_heatmap_pair(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    left_title: str | None = None,
    right_title: str | None = None,
    suptitle: str | None = None,
    region_col: str = "Region name",
    lobe_col: str = "Lobe",
    figsize: tuple[float, float] | None = None,
    cbar_label: str = "fraction of channels significant",
    cbar_width: float = 0.45,
    wspace: float = 0.06,
    title_pad: float = 0.045,
    panel_letters: tuple[str, str] | None = ("A", "B"),
    style: dict | None = None,
    **heatmap_kwargs,
):
    """Side-by-side regional-difference heatmaps, built with :func:`compose_panels`.

    Drop-in replacement for the hand-built
    ``plotting.plot_region_difference_heatmap_pair``: a ``left | right | cbar``
    mosaic where each heatmap is a renderer wrapping
    :func:`~pesco.experimental.plotting.plot_region_difference_heatmap`. Row
    order is locked across panels via ``reference_df = concat(left, right)``;
    the right panel hides its y-ticks and a single colorbar is drawn into the
    spare cell. Extra ``**heatmap_kwargs`` pass through to the heatmap renderer
    (``annot_fontsize``, ``tick_fontsize``, ``facecolor`` …). Returns
    ``(fig, axd)``; the caller saves.
    """
    reference_df = pd.concat([left_df, right_df], ignore_index=True)

    def _n_intervals(df: pd.DataFrame) -> int:
        return df[["interval", "interval_left"]].drop_duplicates().shape[0]

    n_left, n_right = _n_intervals(left_df), _n_intervals(right_df)
    n_rows = len(
        _build_region_rows_meta(
            _prepare_region_df(reference_df, region_col, lobe_col), _LOBE_ORDER
        )
    )
    if figsize is None:
        figsize = (0.5 * (n_left + n_right) + 6.0, max(8.0, 0.38 * n_rows + 2.0))

    def _panel(df, title, show_yticks):
        def draw(ax):
            plot_region_difference_heatmap(
                df, ax=ax, reference_df=reference_df, cbar=False,
                show_yticks=show_yticks, show_ylabel=False, title=title,
                show=False, region_col=region_col, lobe_col=lobe_col,
                **heatmap_kwargs,
            )
        return draw

    fig, axd = compose_panels(
        [["left", "right", "cbar"]],
        {
            "left": _panel(left_df, left_title, True),
            "right": _panel(right_df, right_title, False),
        },
        figsize=figsize,
        width_ratios=[n_left, n_right, cbar_width],
        gridspec_kw={"wspace": wspace},
        sharey_groups=[["left", "right"]],
        titles=[(suptitle, ["left", "right"])] if suptitle else None,
        title_pad=title_pad,
        # heatmaps have wide region-name labels on the left, so the letter goes
        # at the plot-area corner rather than out in the y-tick margin
        letters=(
            [(panel_letters[0], "left"), (panel_letters[1], "right")]
            if panel_letters else None
        ),
        letter_format="{}",
        letter_offset=(2, 8),
        letter_offset_shared=(2, 8),
        style=style,
    )
    # one shared colorbar from the right heatmap's mesh, into the spare cell
    mesh = next(c for c in axd["right"].collections if isinstance(c, QuadMesh))
    fig.colorbar(mesh, cax=axd["cbar"], label=cbar_label)
    return fig, axd


def overlap_heatmap_pair(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    region_lobe: dict,
    *,
    left_grey: pd.DataFrame | None = None,
    left_dot: pd.DataFrame | None = None,
    right_grey: pd.DataFrame | None = None,
    right_dot: pd.DataFrame | None = None,
    left_title: str | None = None,
    right_title: str | None = None,
    suptitle: str | None = None,
    left_xlabel: str = "Frequency band",
    right_xlabel: str = "Frauscher interval (Hz)",
    cbar_label: str = "Overlap",
    cbar_width: float = 0.4,
    wspace: float = 0.06,
    figsize: tuple[float, float] | None = None,
    panel_letters: tuple[str, str] | None = ("A", "B"),
    style: dict | None = None,
    **heatmap_kwargs,
):
    """Two overlap heatmaps side by side (e.g. 5 canonical bands | 22 Frauscher bins).

    A ``left | right | cbar`` mosaic built with :func:`compose_panels`, each
    panel wrapping :func:`~pesco.experimental.plotting.plot_overlap_frauscher_heatmap`.
    The two panels share the region rows (same lobe order, so the y-axes align);
    the right panel hides its y-ticks and a single fixed 0-1 colorbar is drawn
    into the spare cell. ``width_ratios`` follow the band counts, so an
    asymmetric split (e.g. 5 vs 22 columns) is laid out to scale. ``left_*`` /
    ``right_*`` give each panel its own grey/dot masks and title; extra
    ``**heatmap_kwargs`` pass through to both renderers. Returns ``(fig, axd)``;
    the caller saves.
    """
    n_left = left_df.shape[1]
    n_right = right_df.shape[1]
    n_rows = max(left_df.shape[0], right_df.shape[0])
    if figsize is None:
        figsize = (0.42 * (n_left + n_right) + 6.0, max(8.0, 0.32 * n_rows + 2.0))

    def _panel(df, grey, dot, title, xlabel, show_yticks):
        def draw(ax):
            plot_overlap_frauscher_heatmap(
                df,
                region_lobe,
                ax=ax,
                grey=grey,
                dot=dot,
                cbar=False,
                show_yticks=show_yticks,
                show_ylabel=show_yticks,
                title=title,
                xlabel=xlabel,
                **heatmap_kwargs,
            )

        return draw

    fig, axd = compose_panels(
        [["left", "right", "cbar"]],
        {
            "left": _panel(left_df, left_grey, left_dot, left_title, left_xlabel, True),
            "right": _panel(right_df, right_grey, right_dot, right_title, right_xlabel, False),
        },
        figsize=figsize,
        width_ratios=[n_left, n_right, cbar_width],
        gridspec_kw={"wspace": wspace},
        sharey_groups=[["left", "right"]],
        titles=[(suptitle, ["left", "right"])] if suptitle else None,
        letters=(
            [(panel_letters[0], "left"), (panel_letters[1], "right")]
            if panel_letters
            else None
        ),
        letter_format="{}",
        letter_offset=(2, 8),
        letter_offset_shared=(2, 8),
        style=style,
    )
    mesh = next(c for c in axd["left"].collections if isinstance(c, QuadMesh))
    fig.colorbar(mesh, cax=axd["cbar"], label=cbar_label)
    return fig, axd
