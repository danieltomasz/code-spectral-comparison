"""IRASA analysis utilities for the spectral-comparison project.

Mirrors the structure of ``pesco.spectral`` but wraps the ``pyrasa`` library
instead of ``specparam``.  The public API consists of four functions:

* :func:`run_irasa` — thin wrapper around ``pyrasa.irasa``
* :func:`irasa2pandas` — convert ``IrasaSpectrum`` → tidy long-format DataFrame
* :func:`inspect_irasa_fit_quality` — diagnostic histograms (mirrors ``inspect_fit_quality``)
* :func:`inspect_irasa_fits` — plot raw / aperiodic / periodic components
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyrasa import irasa as _pyrasa_irasa
from pyrasa.utils.irasa_spectrum import IrasaSpectrum

FloatArray = NDArray[np.float64]

# ── Project-standard IRASA defaults (match specparam NB 02 Welch settings) ──
# Band upper limit constraint: band[1] * h_max < fs / 2 (Nyquist).
# With fs=200 Hz and h_max=2.0: band[1] < 50 Hz.
# Using (2.0, 40.0) satisfies this (40 * 2.0 = 80 Hz < 100 Hz) and matches
# the robustness range tested in NB 07.
_IRASA_DEFAULTS: dict = dict(
    band=(2.0, 40.0),
    nperseg=400,      # 2 s @ 200 Hz
    noverlap=200,     # 1 s overlap
    hset_info=(1.05, 2.0, 0.05),
    detrend="constant",
    scaling="density",
)

_PEAK_DEFAULTS: dict = dict(
    peak_threshold=2.5,
    min_peak_height=0.0,
    peak_width_limits=(0.5, 12.0),
    smoothing_window=1,
)


def run_irasa(
    data: FloatArray,
    fs: int = 200,
    band: tuple[float, float] = (2.0, 40.0),
    nperseg: int = 400,
    noverlap: int = 200,
    hset_info: tuple[float, float, float] = (1.05, 2.0, 0.05),
    ch_names: np.ndarray | None = None,
) -> IrasaSpectrum:
    """Run IRASA on multi-channel time series data.

    Thin wrapper around :func:`pyrasa.irasa` that enforces the project-standard
    settings (2–100 Hz, 2 s segments, 1 s overlap).

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Raw time-series data. Must be arranged as (Channels, Samples).
    fs : int, default 200
        Sampling frequency in Hz.
    band : tuple of (float, float), default (2.0, 40.0)
        Frequency band for spectral estimation.
        **Constraint**: ``band[1] * h_max < fs / 2`` (Nyquist).
        With the default ``hset_info`` (h_max = 2.0) and ``fs = 200``,
        the upper limit must be < 50 Hz.  Use ``(2.0, 40.0)`` for safety.
    nperseg : int, default 400
        Welch segment length in samples. 400 = 2 s @ 200 Hz.
    noverlap : int, default 200
        Overlap between segments in samples. 200 = 1 s.
    hset_info : tuple of (float, float, float), default (1.05, 2.0, 0.05)
        ``(h_min, h_max, h_step)`` resampling factors for IRASA.
        The default generates 20 h-values (1.05, 1.10, …, 2.0).
    ch_names : array-like or None, default None
        Optional channel name labels; passed through to pyrasa.

    Returns
    -------
    IrasaSpectrum
        pyrasa result object with attributes ``freqs``, ``raw_spectrum``,
        ``aperiodic``, and ``periodic``.  Call
        ``.fit_aperiodic_model('fixed')`` and ``.get_peaks(...)`` to extract
        parameters.

    Notes
    -----
    Peak memory is approximately ``n_h_values × data.nbytes`` where
    ``n_h_values = len(np.arange(h_min, h_max + h_step, h_step))``.
    For the default settings on the Frauscher 2018 dataset (1772 × 13600
    float64) this is about 3.8 GB.  Process in batches if memory is limited.
    """
    return _pyrasa_irasa(
        data=data,
        fs=fs,
        band=band,
        nperseg=nperseg,
        noverlap=noverlap,
        ch_names=ch_names,
        hset_info=hset_info,
        detrend="constant",
        scaling="density",
    )


def irasa2pandas(
    result: IrasaSpectrum,
    fit_func: str = "fixed",
    peak_threshold: float = 2.5,
    min_peak_height: float = 0.0,
    peak_width_limits: tuple[float, float] = (0.5, 12.0),
    smoothing_window: float | int = 1,
) -> pd.DataFrame:
    """Convert an ``IrasaSpectrum`` result to a tidy long-format DataFrame.

    Mirrors :func:`pesco.spectral.specparam2pandas`.  Returns one row per
    detected peak; channels with no peaks are retained with NaN peak columns.
    Aperiodic parameters and goodness-of-fit metrics are repeated for every
    peak belonging to the same channel.

    Parameters
    ----------
    result : IrasaSpectrum
        Output of :func:`run_irasa` (or ``pyrasa.irasa`` called directly).
    fit_func : {"fixed", "knee"}, default "fixed"
        Aperiodic model type.  ``"fixed"`` yields offset + exponent;
        ``"knee"`` adds a knee parameter.  Should match the specparam
        ``aperiodic_mode`` used in the parallel specparam notebook.
    peak_threshold : float, default 2.5
        Relative peak-height threshold (multiples of spectrum SD).
    min_peak_height : float, default 0.0
        Absolute minimum peak height.
    peak_width_limits : tuple of (float, float), default (0.5, 12.0)
        ``(min_width, max_width)`` in Hz.
    smoothing_window : float or int, default 1
        Savitzky-Golay smoothing window (Hz) applied before peak detection.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame.  Columns:

        ``ID``
            int, 0-based channel index (matches channel metadata tables).
        ``offset``
            float, aperiodic offset (log10 units).
        ``exponent``
            float, aperiodic exponent.
        ``error_mae``
            float, mean absolute error of the aperiodic fit.
        ``gof_rsquared``
            float, R² of the aperiodic fit.
        ``cf``
            float or NaN, peak centre frequency (Hz).
        ``pw``
            float or NaN, peak height (log10 power).
        ``bw``
            float or NaN, peak bandwidth (Hz).

    Notes
    -----
    Column names use lowercase ``cf / pw / bw`` (pyrasa convention), which is
    intentionally different from specparam's ``CF / PW / BW``, so that
    columns from the two methods cannot collide silently when DataFrames are
    merged side-by-side in the comparison notebook.

    ``ID`` is created positionally from ``range(n_channels)``, independent of
    whether ``ch_names`` was passed to :func:`run_irasa`.  This makes it
    safe to join directly with channel-metadata tables that use the same
    positional convention.
    """
    # ── Fit aperiodic model ──────────────────────────────────────────────────
    ap_fit = result.fit_aperiodic_model(fit_func)

    # aperiodic_params has columns Offset, Exponent (and optionally Knee)
    ap_df = ap_fit.aperiodic_params[["Offset", "Exponent"]].copy()
    ap_df.columns = ["offset", "exponent"]
    ap_df["ID"] = range(len(ap_df))

    # gof has columns R2, mse (mean squared error — we rename to mae for
    # consistency with specparam naming; note: pyrasa reports MSE not MAE)
    gof_df = ap_fit.gof[["R2", "mse"]].copy()
    gof_df.columns = ["gof_rsquared", "error_mae"]
    gof_df["ID"] = range(len(gof_df))

    aperiodic_df = ap_df.merge(gof_df, on="ID")

    # ── Extract peaks ────────────────────────────────────────────────────────
    peaks_df = result.get_peaks(
        peak_threshold=peak_threshold,
        min_peak_height=min_peak_height,
        peak_width_limits=peak_width_limits,
        smoothing_window=smoothing_window,
    )

    if peaks_df is not None and len(peaks_df) > 0 and "ch_name" in peaks_df.columns:
        peaks_df = peaks_df[["ch_name", "cf", "pw", "bw"]].copy()
        # ch_name is "0", "1", ... when ch_names=None; use positional ID
        # by resetting to match the row order in aperiodic_df
        peaks_df["ID"] = pd.to_numeric(peaks_df["ch_name"], errors="coerce").astype(
            "Int64"
        )
        peaks_df = peaks_df.drop(columns="ch_name")
    else:
        peaks_df = pd.DataFrame(columns=["ID", "cf", "pw", "bw"])

    # ── Left-join: keep all channels even those with no peaks ───────────────
    result_df = aperiodic_df.merge(peaks_df, on="ID", how="left")
    return result_df


def fit_irasa_slope(
    freqs: FloatArray, psd_fractal: FloatArray
) -> tuple[float, float]:
    """Fit a log-log linear slope to an IRASA aperiodic spectrum.

    Useful when you have a pre-computed aperiodic PSD (e.g. from a manual
    IRASA pass or from a single region) and want to extract the exponent and
    offset without going through the full ``IrasaSpectrum`` workflow.

    Parameters
    ----------
    freqs : array
        Frequency values. Only values > 0 are used.
    psd_fractal : array
        Aperiodic (fractal) PSD values from IRASA decomposition. Must be the
        same length as ``freqs``. Values are clipped to 1e-30 before log
        transform to avoid -inf.

    Returns
    -------
    exponent : float
        Aperiodic exponent (positive; 1/f-like spectra have exponent > 0).
    intercept : float
        Log-log intercept (equivalent to aperiodic offset).
    """
    mask = freqs > 0
    log_f = np.log10(freqs[mask])
    log_p = np.log10(np.clip(psd_fractal[mask], 1e-30, None))
    slope, intercept = np.polyfit(log_f, log_p, 1)
    return -slope, intercept  # positive exponent convention


def inspect_irasa_fit_quality(
    results_df: pd.DataFrame,
    r2_threshold: float = 0.80,
    bins: int = 40,
    show: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure, np.ndarray]:
    """Print fit-quality summary statistics and plot diagnostic histograms.

    Mirrors :func:`pesco.spectral.inspect_fit_quality`.  Operates on the
    DataFrame returned by :func:`irasa2pandas`.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Long-format DataFrame returned by :func:`irasa2pandas`.
    r2_threshold : float, default 0.80
        R² threshold used to compute and report the fraction of good-fit channels.
    bins : int, default 40
        Number of histogram bins.
    show : bool, default True
        If True, call ``plt.show()`` at the end.

    Returns
    -------
    ch_df : pandas.DataFrame
        Per-channel summary (one row per channel, peak columns dropped).
    summary_df : pandas.DataFrame
        Descriptive statistics for exponent, R², and MAE.
    fig : matplotlib.Figure
    axes : np.ndarray of Axes
        Three-panel figure: exponent distribution, R² distribution, MAE distribution.
    """
    required = {"ID", "exponent", "offset", "gof_rsquared", "error_mae"}
    missing = sorted(required - set(results_df.columns))
    if missing:
        raise ValueError(f"results_df is missing required columns: {', '.join(missing)}")

    ch_df = (
        results_df.drop_duplicates(subset="ID")[
            ["ID", "exponent", "offset", "gof_rsquared", "error_mae"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    summary_df = ch_df[["exponent", "gof_rsquared", "error_mae"]].describe().round(3)
    print("IRASA fit quality summary")
    print("=" * 40)
    print(summary_df.to_string())

    n_good = (ch_df["gof_rsquared"] >= r2_threshold).sum()
    total = len(ch_df)
    pct = 100.0 * n_good / total if total else float("nan")
    print(f"\nChannels with R² ≥ {r2_threshold}: {n_good} / {total} ({pct:.1f} %)")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))

    axes[0].hist(ch_df["exponent"].dropna(), bins=bins, edgecolor="white")
    axes[0].set_xlabel("Exponent")
    axes[0].set_title("Exponent distribution")

    axes[1].hist(ch_df["gof_rsquared"].dropna(), bins=bins, edgecolor="white")
    axes[1].axvline(
        r2_threshold, color="red", linestyle="--", label=f"threshold = {r2_threshold}"
    )
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("R²")
    axes[1].set_title("R² distribution")

    axes[2].hist(ch_df["error_mae"].dropna(), bins=bins, edgecolor="white")
    axes[2].set_xlabel("MSE (aperiodic fit)")
    axes[2].set_title("MSE distribution")

    fig.suptitle("IRASA — aperiodic fit quality", fontsize=12)
    fig.tight_layout()
    if show:
        plt.show()

    return ch_df, summary_df, fig, axes


def inspect_irasa_fits(
    result: IrasaSpectrum,
    results_df: pd.DataFrame | None = None,
    select_by: Literal["exponent", "gof"] = "exponent",
    n_spectra: int = 5,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot representative raw, aperiodic, and periodic spectra.

    Mirrors :func:`pesco.spectral.inspect_fits` but exploits the three-way
    IRASA decomposition.  For each selected channel the raw spectrum,
    empirical aperiodic component, and periodic (oscillatory) component are
    shown as separate lines.

    Parameters
    ----------
    result : IrasaSpectrum
        Output of :func:`run_irasa`.
    results_df : pandas.DataFrame or None, default None
        Per-channel summary from :func:`irasa2pandas`.  Required when
        ``select_by`` is ``"exponent"`` or ``"gof"``; if None, channels are
        selected by position (equally spaced across channel indices).
    select_by : {"exponent", "gof"}, default "exponent"
        Criterion for choosing representative channels.
    n_spectra : int, default 5
        Number of channels to display.
    show : bool, default True
        If True, call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.Figure
    axes : np.ndarray of Axes
        One subplot per selected channel (1 row × n_spectra columns).
    """
    freqs = result.freqs
    raw = result.raw_spectrum      # (n_channels, n_freqs)
    aperiodic = result.aperiodic   # (n_channels, n_freqs)
    periodic = result.periodic     # (n_channels, n_freqs)
    n_ch = raw.shape[0]

    # ── Select channel indices ───────────────────────────────────────────────
    if results_df is not None and select_by in ("exponent", "gof"):
        ch_df = results_df.drop_duplicates(subset="ID").copy()
        metric = "exponent" if select_by == "exponent" else "gof_rsquared"
        sorted_ids = ch_df.sort_values(metric)["ID"].values
        quantile_positions = np.linspace(0, len(sorted_ids) - 1, n_spectra, dtype=int)
        selected = sorted_ids[quantile_positions]
    else:
        selected = np.linspace(0, n_ch - 1, n_spectra, dtype=int)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_spectra, figsize=(4 * n_spectra, 4), sharey=False)
    if n_spectra == 1:
        axes = np.array([axes])

    for ax, ch_idx in zip(axes, selected):
        ax.semilogy(freqs, raw[ch_idx], color="black", linewidth=1.0, label="raw")
        ax.semilogy(
            freqs, aperiodic[ch_idx], color="tomato", linewidth=1.5, label="aperiodic"
        )
        # Periodic component can be negative — clip to small positive for log plot
        periodic_plot = np.clip(periodic[ch_idx], a_min=1e-15, a_max=None)
        ax.semilogy(
            freqs,
            periodic_plot,
            color="steelblue",
            linewidth=1.0,
            linestyle="--",
            label="periodic",
        )

        exp_str = ""
        if results_df is not None:
            row = results_df[results_df["ID"] == ch_idx]
            if len(row):
                exp_val = row["exponent"].iloc[0]
                r2_val = row["gof_rsquared"].iloc[0]
                exp_str = f"\nexp={exp_val:.2f}, R²={r2_val:.2f}"
        ax.set_title(f"CH {ch_idx}{exp_str}", fontsize=9)
        ax.set_xlabel("Frequency (Hz)")
        if ax is axes[0]:
            ax.set_ylabel("PSD (V²/Hz)")
            ax.legend(fontsize=7)

    fig.suptitle(
        f"IRASA — representative spectra (selected by {select_by})", fontsize=11
    )
    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes
