from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

if TYPE_CHECKING:
    from specparam import SpectralGroupModel

FloatArray: TypeAlias = NDArray[np.float64]
AxesArray: TypeAlias = NDArray[np.object_]
QStatus: TypeAlias = Literal["unavailable", "clear_doubt", "mild_concern", "adequate"]


def aperiodic_curve(
    offset: float | FloatArray,
    exponent: float | FloatArray,
    freqs: FloatArray,
) -> FloatArray:
    """Reconstruct the aperiodic component in log10-power space.

    Computes ``offset - exponent * log10(freqs)``, i.e. the fixed-mode
    specparam model evaluated at ``freqs``.  Useful for subtracting the
    aperiodic background from a log-PSD or for overlaying it on a log-scale plot.

    Parameters
    ----------
    offset : float or array of shape (N,)
        Aperiodic offset parameter(s).
    exponent : float or array of shape (N,)
        Aperiodic exponent parameter(s).
    freqs : array of shape (F,)
        Frequency vector (Hz). Must be strictly positive.

    Returns
    -------
    numpy.ndarray
        Shape ``(F,)`` when ``offset``/``exponent`` are scalars, or
        ``(N, F)`` when they are arrays of length N.
    """
    freqs = np.asarray(freqs, dtype=float)
    offset = np.asarray(offset, dtype=float)
    exponent = np.asarray(exponent, dtype=float)

    log_f = np.log10(freqs)  # shape (F,)

    if offset.ndim == 0 and exponent.ndim == 0:
        # scalar case → (F,)
        return float(offset) - float(exponent) * log_f

    # vectorised case → (N, F)
    return offset[:, np.newaxis] - exponent[:, np.newaxis] * log_f[np.newaxis, :]


# Inspect example spectra: low / median / high exponent, with aperiodic overlay
def _aperiodic_curve(
    freqs: FloatArray,
    offset: float,
    exponent: float,
    knee: float | None = None,
) -> FloatArray:
    if knee is None or np.isnan(knee):
        return 10**offset / (freqs**exponent)
    return 10**offset / (knee + freqs**exponent)


def _extract_fit_metric(
    fg: SpectralGroupModel,
    metric_group: str,
    metric_name: str,
    metric_key: str,
) -> FloatArray:
    """Return fit metrics, falling back to result-level metrics if needed."""
    metrics = np.asarray(fg.get_metrics(metric_group, metric_name), dtype=float)

    if metrics.size == 0 or np.all(np.isnan(metrics)):
        group_res = fg.results.group_results
        return np.array([r.metrics.get(metric_key, np.nan) for r in group_res], dtype=float)

    return metrics


def _classify_q(q: float, q_mild: float, q_doubt: float) -> QStatus:
    """Classify residual-curvature diagnostic values using notebook thresholds."""
    if np.isnan(q):
        return "unavailable"
    if abs(q) > q_doubt:
        return "clear_doubt"
    if abs(q) > q_mild:
        return "mild_concern"
    return "adequate"


def compute_curvature_q(
    fg: SpectralGroupModel,
    peak_rm: bool = True,
    q_mild: float = 0.10,
    q_doubt: float = 0.30,
) -> pd.DataFrame:
    """Compute the quadratic residual-curvature diagnostic for each fitted spectrum.

    Parameters
    ----------
    fg : SpectralGroupModel
        A fitted group model, typically in fixed aperiodic mode.
    peak_rm : bool, optional, default: True
        If True, compute residuals from the peak-removed spectrum minus the
        aperiodic fit. If False, use raw log-power minus the aperiodic fit.
    q_mild : float, optional, default: 0.10
        Threshold for mild concern.
    q_doubt : float, optional, default: 0.30
        Threshold for clear doubt.

    Returns
    -------
    pandas.DataFrame
        One row per fitted spectrum with columns:
        `ID`, `q`, `q_b`, `q_c`, `q_abs`, `q_status`.
    """
    if not fg.results.has_model:
        raise ValueError("No model fit results available. Please fit the model first.")

    rows: list[dict[str, Any]] = []
    for idx, _ in enumerate(fg.results.group_results):
        sm = fg.get_model(idx, regenerate=True)
        x = np.log10(sm.data.freqs)
        ap_fit = np.asarray(sm.results.model._ap_fit, dtype=float)
        log_power = np.asarray(sm.data.power_spectrum, dtype=float)

        # Regenerated group models expose the peak fit reliably in rc6, even though
        # `_spectrum_peak_rm` itself is not regenerated.
        peak_fit = sm.results.model._peak_fit
        if peak_rm and peak_fit is not None:
            residuals = log_power - np.asarray(peak_fit, dtype=float) - ap_fit
        else:
            residuals = log_power - ap_fit

        valid = np.isfinite(x) & np.isfinite(residuals)
        if valid.sum() < 3:
            coeffs = np.array([np.nan, np.nan, np.nan], dtype=float)
        else:
            coeffs = np.polyfit(x[valid], residuals[valid], 2)

        q = coeffs[0]
        rows.append(
            {
                "ID": idx,
                "q": q,
                "q_b": coeffs[1],
                "q_c": coeffs[2],
                "q_abs": abs(q) if np.isfinite(q) else np.nan,
                "q_status": _classify_q(q, q_mild, q_doubt),
            }
        )

    return pd.DataFrame(rows)


def inspect_fits(
    fg: SpectralGroupModel,
    psds_fit: FloatArray,
    freqs: FloatArray,
    select_by: Literal["exponent", "gof", "rsquared"] = "exponent",
    n_spectra: Literal[3, 5] = 5,
) -> tuple[Figure, Axes]:
    """Plot representative spectra with fitted aperiodic overlay.

    Parameters
    ----------
    select_by : {"exponent", "gof", "rsquared"}
        Select example spectra by low / median / high exponent or goodness of fit.
    n_spectra : {3, 5}
        Number of representative spectra to show.
    """
    group_res = fg.results.group_results
    rsquared = _extract_fit_metric(fg, "gof", "rsquared", "gof_rsquared")
    exps = np.array([r.aperiodic_fit[-1] for r in group_res])

    if select_by == "exponent":
        selector = exps
        selector_label = "exponent"
    elif select_by in {"gof", "rsquared"}:
        selector = rsquared
        selector_label = "goodness of fit"
    else:
        raise ValueError("select_by must be one of: 'exponent', 'gof', 'rsquared'")

    sorted_ids = np.argsort(selector)
    if n_spectra == 3:
        quantiles = [0.0, 0.5, 1.0]
        colors = ["C0", "C1", "C2"]
        labels = ["Low", "Median", "High"]
    elif n_spectra == 5:
        quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
        colors = ["C0", "C1", "C2", "C3", "C4"]
        labels = ["Low", "Q25", "Median", "Q75", "High"]
    else:
        raise ValueError("n_spectra must be either 3 or 5")

    quantile_positions = np.rint(
        np.quantile(np.arange(len(sorted_ids)), quantiles)
    ).astype(int)
    ids_to_show = sorted_ids[quantile_positions]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, color, label in zip(ids_to_show, colors, labels):
        r = group_res[idx]
        offset   = r.aperiodic_fit[0]
        exponent = r.aperiodic_fit[-1]
        ax.semilogy(freqs, psds_fit[idx], color=color, alpha=0.7,
                    label=f"{label} exp={exponent:.2f} (R²={rsquared[idx]:.2f})")
        ap = _aperiodic_curve(fg.data.freqs, offset, exponent)
        ax.semilogy(fg.data.freqs, ap, color=color, linestyle="--", linewidth=2)

    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD")
    ax.set_title(
        f"Example spectra selected by {selector_label} "
        "(dashed = fitted aperiodic)"
    )
    ax.legend()
    plt.tight_layout()
    return fig, ax


def inspect_fit_quality(
    results_df: pd.DataFrame,
    r2_threshold: float = 0.80,
    bins: int = 40,
    show: bool = True,
    q_mild: float = 0.10,
    q_doubt: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, Figure, AxesArray]:
    """Print fit-quality summary statistics and plot diagnostic histograms.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Long-format DataFrame returned by `specparam2pandas`.
        If q columns are present, they are included automatically in the
        summary outputs and diagnostic plots.
    """
    results_df = results_df.copy()

    required_columns = {"ID", "error_mae", "gof_rsquared", "exponent", "offset"}
    missing_columns = sorted(required_columns - set(results_df.columns))
    if missing_columns:
        raise ValueError(
            "results_df is missing required columns: "
            + ", ".join(missing_columns)
        )

    has_q = "q" in results_df.columns
    if has_q and "q_abs" not in results_df.columns:
        results_df["q_abs"] = results_df["q"].abs()
    if has_q and "q_status" not in results_df.columns:
        results_df["q_status"] = results_df["q"].apply(
            lambda q: _classify_q(q, q_mild, q_doubt)
        )

    summary_columns = ["error_mae", "gof_rsquared", "exponent"]
    ch_columns = ["ID", "error_mae", "gof_rsquared", "exponent", "offset"]
    if has_q:
        ch_columns.extend(["q", "q_abs", "q_status"])
        summary_columns.extend(["q", "q_abs"])

    ch_df = results_df.drop_duplicates(subset="ID")[ch_columns].copy()

    summary_df = ch_df[summary_columns].describe().round(3)
    print("Fit quality summary:")
    print(summary_df)

    n_good = (ch_df["gof_rsquared"] >= r2_threshold).sum()
    total = len(ch_df)
    pct_good = 100 * n_good / total if total else np.nan
    print(
        f"\nChannels with R\u00b2 \u2265 {r2_threshold}: "
        f"{n_good} / {total} ({pct_good:.1f}%)"
    )

    if has_q:
        print("\nq status:")
        print(ch_df["q_status"].value_counts())

    n_panels = 4 if has_q else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))
    axes = np.atleast_1d(axes)

    axes[0].hist(ch_df["exponent"].dropna(), bins=bins, edgecolor="white")
    axes[0].set_xlabel("Exponent")
    axes[0].set_title("Exponent distribution")

    axes[1].hist(ch_df["gof_rsquared"].dropna(), bins=bins, edgecolor="white")
    axes[1].axvline(
        r2_threshold,
        color="red",
        linestyle="--",
        label=f"threshold={r2_threshold}",
    )
    axes[1].legend()
    axes[1].set_xlabel("R²")
    axes[1].set_title("R² distribution")

    axes[2].hist(ch_df["error_mae"].dropna(), bins=bins, edgecolor="white")
    axes[2].set_xlabel("MAE")
    axes[2].set_title("MAE distribution")

    if has_q:
        axes[3].hist(ch_df["q"].dropna(), bins=bins, edgecolor="white")
        axes[3].axvline(q_mild, color="orange", linestyle="--", label=f"mild={q_mild}")
        axes[3].axvline(-q_mild, color="orange", linestyle="--")
        axes[3].axvline(q_doubt, color="red", linestyle="--", label=f"doubt={q_doubt}")
        axes[3].axvline(-q_doubt, color="red", linestyle="--")
        axes[3].legend()
        axes[3].set_xlabel("q")
        axes[3].set_title("Curvature distribution")

    fig.tight_layout()

    if show:
        plt.show()

    return ch_df, summary_df, fig, axes


def inspect_q_extremes(
    results_df: pd.DataFrame,
    fg: SpectralGroupModel,
    psds_fit: FloatArray,
    freqs: FloatArray,
    n_extremes: int = 10,
    show: bool = True,
    q_mild: float = 0.10,
    q_doubt: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, Figure, AxesArray]:
    """Plot spectra with the lowest and highest q values.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Long-format DataFrame returned by `specparam2pandas`, including q columns.
    fg : SpectralGroupModel
        Fitted group model used to recover the aperiodic fits.
    psds_fit : array
        Power spectra used for fitting, restricted to the fitted frequency range.
    freqs : array
        Frequency vector corresponding to `psds_fit`.
    n_extremes : int, optional, default: 10
        Number of lowest-q and highest-q spectra to plot.

    Returns
    -------
    low_df, high_df, fig, axes
        Selected low-q summary, selected high-q summary, and the figure / axes.
    """
    required_columns = {"ID", "q", "gof_rsquared", "offset", "exponent"}
    missing_columns = sorted(required_columns - set(results_df.columns))
    if missing_columns:
        raise ValueError(
            "results_df is missing required columns for q inspection: "
            + ", ".join(missing_columns)
        )
    if n_extremes < 1:
        raise ValueError("n_extremes must be at least 1")

    ch_df = results_df.drop_duplicates(subset="ID").copy()
    ch_df = ch_df[np.isfinite(ch_df["q"])].copy()
    if ch_df.empty:
        raise ValueError("results_df does not contain any finite q values to inspect.")

    if "q_abs" not in ch_df.columns:
        ch_df["q_abs"] = ch_df["q"].abs()
    if "q_status" not in ch_df.columns:
        ch_df["q_status"] = ch_df["q"].apply(lambda q: _classify_q(q, q_mild, q_doubt))

    n_select = min(n_extremes, len(ch_df))
    low_df = ch_df.nsmallest(n_select, "q").reset_index(drop=True)
    high_df = ch_df.nlargest(n_select, "q").reset_index(drop=True)

    fig, axes = plt.subplots(
        n_select, 2, figsize=(12, max(2.6 * n_select, 4)), sharex=True, squeeze=False
    )

    def _status_color(q_status: str) -> str:
        if q_status == "clear_doubt":
            return "crimson"
        if q_status == "mild_concern":
            return "darkorange"
        return "black"

    def _plot_selected(ax: Axes, row: pd.Series, line_color: str, column_title: str) -> None:
        idx = int(row["ID"])
        ap = _aperiodic_curve(freqs, float(row["offset"]), float(row["exponent"]))
        ax.semilogy(freqs, psds_fit[idx], color=line_color, alpha=0.8, linewidth=1.4)
        ax.semilogy(freqs, ap, color=line_color, linestyle="--", linewidth=2.0)
        ax.set_title(
            (
                f"{column_title} | ID {idx} | q={row['q']:.2f} | "
                f"R²={row['gof_rsquared']:.2f} | exp={row['exponent']:.2f}\n"
                f"{row['q_status']}"
            ),
            fontsize=9,
            color=_status_color(str(row["q_status"])),
        )
        ax.grid(alpha=0.2, linewidth=0.5)

    for row_idx in range(n_select):
        _plot_selected(axes[row_idx, 0], low_df.iloc[row_idx], "C0", "Lowest q")
        _plot_selected(axes[row_idx, 1], high_df.iloc[row_idx], "C3", "Highest q")

        if row_idx == 0:
            axes[row_idx, 0].legend(
                ["spectrum", "aperiodic fit"], fontsize=8, loc="best", frameon=False
            )
            axes[row_idx, 1].legend(
                ["spectrum", "aperiodic fit"], fontsize=8, loc="best", frameon=False
            )

    axes[-1, 0].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("PSD")

    fig.suptitle(
        "Extreme q spectra\n"
        "Left: most negative q (knee-like curvature) | "
        "Right: most positive q (upward / floor-like curvature)",
        y=1.01,
    )
    fig.tight_layout()

    if show:
        plt.show()

    return low_df, high_df, fig, axes


def specparam2pandas(
    fg: SpectralGroupModel,
    add_q: bool = False,
    peak_rm: bool = True,
    q_mild: float = 0.10,
    q_doubt: float = 0.30,
) -> pd.DataFrame:
    """
    Converts a SpectralGroupModel object into a pandas DataFrame, with peak parameters and
    corresponding aperiodic fit information.

    Args:
    -----
    fg : SpectralGroupModel
        The SpectralGroupModel object containing the fitting results.
    add_q : bool, optional, default: False
        If True, compute and merge the quadratic residual-curvature diagnostic
        from `compute_curvature_q`.
    peak_rm : bool, optional, default: True
        Passed to `compute_curvature_q` when `add_q=True`.
        If True, q is computed from the peak-removed spectrum minus the aperiodic fit.
        If False, q is computed from the raw log-power spectrum minus the aperiodic fit.
    q_mild : float, optional, default: 0.10
        Mild-concern threshold used to classify q when `add_q=True`.
    q_doubt : float, optional, default: 0.30
        Clear-doubt threshold used to classify q when `add_q=True`.

    Returns:
    --------
    result : pandas.DataFrame
        A DataFrame with the peak parameters and corresponding aperiodic fit information.
        Each row represents a single peak, with columns:
        - 'CF': center frequency of each peak
        - 'PW': power of each peak
        - 'BW': bandwidth of each peak
        - 'ID': identifier for the spectrum this peak belongs to
        - Aperiodic parameters: 'offset', 'exponent' (and 'knee' if knee mode)
        - 'error_mae': mean absolute error of the fit
        - 'gof_rsquared': R-squared value of the fit
        - Optional q diagnostic columns: 'q', 'q_b', 'q_c', 'q_abs', 'q_status'

    Notes:
    ------
    This function creates a long-format DataFrame where each peak is a separate row.
    Peaks are joined with their corresponding aperiodic parameters via the 'ID' column.
    If a spectrum has no peaks, it will still appear with NaN values for peak columns.
    When `add_q=True`, q is computed per spectrum by fitting a quadratic to residuals in
    log-log space. For oscillatory spectra, the recommended setting is `peak_rm=True` so
    q reflects curvature in the aperiodic component rather than curvature introduced by peaks.
    """

    # Check if model has been fit
    if not fg.results.has_model:
        raise ValueError("No model fit results available. Please fit the model first.")

    # Extract aperiodic parameters - one row per spectrum
    ap_params = fg.get_params("aperiodic")
    ap_labels = list(fg.modes.aperiodic.params.labels)

    specparam_aperiodic = pd.DataFrame(ap_params, columns=ap_labels)

    # Prefer the public accessor, but fall back to result-level metrics for rc builds
    # that can silently return all-NaN arrays here.
    specparam_aperiodic["error_mae"] = _extract_fit_metric(
        fg, "error", "mae", "error_mae"
    )
    specparam_aperiodic["gof_rsquared"] = _extract_fit_metric(
        fg, "gof", "rsquared", "gof_rsquared"
    )

    # Add ID column
    specparam_aperiodic = specparam_aperiodic.reset_index(names=["ID"])

    if add_q:
        q_df = compute_curvature_q(
            fg, peak_rm=peak_rm, q_mild=q_mild, q_doubt=q_doubt
        )
        specparam_aperiodic = specparam_aperiodic.merge(q_df, on="ID", how="left")

    # Extract peak parameters
    peaks = fg.get_params("peak")

    if peaks.size > 0:
        # peaks array has shape (n_peaks, 4) where columns are [CF, PW, BW, ID]
        # The last column is the model index
        peak_df = pd.DataFrame(peaks)
        peak_df.columns = ["CF", "PW", "BW", "ID"]
        peak_df["ID"] = peak_df["ID"].astype(int)

        # Left join peaks with aperiodic parameters
        result = specparam_aperiodic.merge(peak_df, on="ID", how="left")
    else:
        # No peaks found - create empty peak dataframe with proper columns
        peak_df = pd.DataFrame(columns=["CF", "PW", "BW", "ID"])

        # Left join to maintain all spectra with NaN for peak values
        result = specparam_aperiodic.merge(peak_df, on="ID", how="left")

    return result
