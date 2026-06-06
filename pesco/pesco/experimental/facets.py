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

from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pesco.experimental.plotting import _plot_subplot, plot_clusters

_CANONICAL_LOBES = ("Occipital", "Parietal", "Frontal", "Temporal")

_JOURNAL_STYLE: dict = {
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
