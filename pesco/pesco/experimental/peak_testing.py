"""Stage 3 of the Frauscher 2018 pipeline: bin into frequency intervals and KS-test against no-peak."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu


# Paper Table: 22 frequency intervals (Hz boundaries)
_PAPER_INTERVALS = (
    0.5, 0.75, 1.25, 1.75, 2.25, 3.25, 3.75, 4.25, 5.25, 6.25, 6.75,
    7.75, 8.25, 9.25, 10.25, 11.75, 13.25, 15.25, 17.25, 20.25, 24.25,
    31.75, 80,
)


def cutintervals(
    x: np.ndarray,
    edges: tuple[float, ...] | np.ndarray | None = None,
) -> tuple[pd.Categorical, np.ndarray]:
    """Bin frequencies into the paper's 22 intervals (or custom edges)."""
    if edges is None:
        edges = _PAPER_INTERVALS
    colBin, y = pd.cut(x, edges, retbins=True, include_lowest=True)
    return colBin, y


def compute_intervals(
    f: np.ndarray,
    psd_reference: np.ndarray,
    threshold: float = 0.04,
) -> np.ndarray:
    """Frauscher-style frequency-interval edges: greedy ≥``threshold`` cuts.

    Each interval holds at least ``threshold`` of the total mass of
    ``psd_reference``. Uses the methodology from Frauscher et al. 2018:
    "The intervals were selected so that each of them had at least 4%
    of the power of the average normalized spectrum of all the studied
    channels".

    Parameters
    ----------
    f : array, shape (n_freqs,)
        Frequency grid (uniformly spaced).
    psd_reference : array, shape (n_freqs,)
        Reference spectrum — typically the mean of the per-channel
        normalized PSDs. Need not sum to 1; it's renormalized internally.
    threshold : float, default 0.04
        Minimum mass per interval, as a fraction of the total.

    Returns
    -------
    edges : array
        Sorted edges. Cuts are placed midway between consecutive ``f``
        bins, matching the paper's 0.5/0.75/1.25/... grid.
    """
    f = np.asarray(f, dtype=float)
    psd_reference = np.asarray(psd_reference, dtype=float)
    df = f[1] - f[0]
    mass = psd_reference / psd_reference.sum()  # normalize to total = 1
    edges = [f[0] - df / 2]
    cum = 0.0
    for i in range(len(f) - 1):
        cum += mass[i]
        if cum >= threshold:
            edges.append(f[i] + df / 2)
            cum = 0.0
    edges.append(f[-1] + df / 2)
    return np.asarray(edges)


def get_intervals(psd: pd.DataFrame, colbin: pd.Categorical) -> pd.DataFrame:
    """Sum PSD power within each frequency interval."""
    dictionary = dict(zip(psd.columns, colbin))
    psd_intervals = psd.T.groupby(dictionary).sum().T
    psd_intervals.columns = psd_intervals.columns.astype(str, copy=False)
    return psd_intervals


def _frequency_columns(df: pd.DataFrame, colbin: pd.Categorical) -> list:
    """Return spectral columns matching ``colbin``, ignoring metadata columns."""
    numeric_name_cols = [
        col
        for col in df.columns
        if isinstance(col, (int, float, np.integer, np.floating))
    ]
    if len(numeric_name_cols) >= len(colbin):
        return numeric_name_cols[:len(colbin)]
    return list(df.columns[:len(colbin)])


def _interval_bounds(label: str) -> tuple[float, float]:
    pattern = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")
    bounds = [float(v) for v in pattern.findall(label)]
    if len(bounds) != 2:
        raise ValueError(f"Could not parse interval bounds from {label!r}")
    return bounds[0], bounds[1]


def _channel_fraction_significant(
    channel_values: np.ndarray,
    no_peak_values: np.ndarray,
    alpha: float,
) -> tuple[int, float]:
    significant = 0
    no_peak_values = np.asarray(no_peak_values, dtype=float)
    if len(no_peak_values) == 0:
        return 0, np.nan
    for value in np.asarray(channel_values, dtype=float):
        res = mannwhitneyu(
            [value],
            no_peak_values,
            alternative="greater",
            method="auto",
        )
        if res.pvalue < alpha:
            significant += 1
    if len(channel_values) == 0:
        return 0, np.nan
    return significant, significant / len(channel_values)


def _ks_and_channel_fraction(
    group_values: np.ndarray,
    no_peak_values: np.ndarray | None,
    n_tests: int,
    alpha: float,
    channel_alpha: float,
) -> dict:
    n_group = len(group_values)
    n_no_peak = 0 if no_peak_values is None else len(no_peak_values)
    out = {
        "ks_statistic": np.nan,
        "ks_pvalue": np.nan,
        "ks_pvalue_corrected": np.nan,
        "ks_significant": False,
        "n_region_channels": n_group,
        "n_no_peak_channels": n_no_peak,
        "n_channel_significant": 0,
        "channel_fraction": np.nan,
    }
    if no_peak_values is None or n_group == 0 or n_no_peak == 0:
        return out
    res = ks_2samp(group_values, no_peak_values, alternative="less", method="asymp")
    out["ks_statistic"] = float(res.statistic)
    out["ks_pvalue"] = float(res.pvalue)
    out["ks_pvalue_corrected"] = min(float(res.pvalue) * n_tests, 1.0)
    out["ks_significant"] = out["ks_pvalue_corrected"] < alpha
    if out["ks_significant"]:
        n_sig, frac = _channel_fraction_significant(
            group_values, no_peak_values, channel_alpha
        )
        out["n_channel_significant"] = n_sig
        out["channel_fraction"] = frac
    return out


def test_regions_heatmap(
    no_peak_df: pd.DataFrame | None,
    psd: pd.DataFrame,
    colbin: pd.Categorical,
    *,
    alpha: float = 0.05,
    channel_alpha: float = 0.05,
    region_col: str = "Region name",
    lobe_col: str = "Lobe",
    include_lobes: bool = True,
) -> pd.DataFrame:
    """Frauscher-style region x frequency-bin tests for heatmap plotting.

    Each region/bin is first screened with a Bonferroni-corrected one-sided
    KS test against the no-peak set. For significant cells, each regional
    channel is tested against the no-peak distribution with a one-sided
    rank-sum/Mann-Whitney test; ``channel_fraction`` is the fraction of
    channels passing that uncorrected second-stage test.

    With ``include_lobes=True`` (default) the result also contains rows
    aggregating all channels of each lobe. Those rows have
    ``Region name == Lobe`` and ``is_lobe_row == True``. Bonferroni
    correction uses ``(n_regions + n_lobes) * n_intervals`` to match the
    Frauscher 2018 convention (38 regions + 4 lobes).
    """
    freq_cols = _frequency_columns(psd, colbin)
    psd_intervals = get_intervals(psd[freq_cols], colbin).assign(
        **{
            lobe_col: psd[lobe_col].to_numpy(),
            region_col: psd[region_col].to_numpy(),
        }
    )
    interval_cols = [
        col for col in psd_intervals.columns if col not in {lobe_col, region_col}
    ]
    regions = psd_intervals[[lobe_col, region_col]].drop_duplicates()
    lobes = psd_intervals[lobe_col].drop_duplicates() if include_lobes else []
    n_groups = len(regions) + (len(lobes) if include_lobes else 0)
    n_tests = max(n_groups * len(interval_cols), 1)

    if no_peak_df is not None:
        no_peak_freq_cols = _frequency_columns(no_peak_df, colbin)
        no_peak_intervals = get_intervals(no_peak_df[no_peak_freq_cols], colbin)
    else:
        no_peak_intervals = None

    rows = []

    if include_lobes:
        for lobe in lobes:
            lobe_intervals = psd_intervals[psd_intervals[lobe_col] == lobe][
                interval_cols
            ]
            for interval in interval_cols:
                left, right = _interval_bounds(interval)
                np_values = (
                    no_peak_intervals[interval].to_numpy(dtype=float)
                    if no_peak_intervals is not None
                    else None
                )
                stats = _ks_and_channel_fraction(
                    lobe_intervals[interval].to_numpy(dtype=float),
                    np_values,
                    n_tests,
                    alpha,
                    channel_alpha,
                )
                rows.append(
                    {
                        lobe_col: lobe,
                        region_col: lobe,
                        "is_lobe_row": True,
                        "interval": interval,
                        "interval_left": left,
                        "interval_right": right,
                        **stats,
                    }
                )

    for _, region_meta in regions.iterrows():
        lobe = region_meta[lobe_col]
        region = region_meta[region_col]
        region_intervals = psd_intervals[
            (psd_intervals[lobe_col] == lobe)
            & (psd_intervals[region_col] == region)
        ][interval_cols]

        for interval in interval_cols:
            left, right = _interval_bounds(interval)
            np_values = (
                no_peak_intervals[interval].to_numpy(dtype=float)
                if no_peak_intervals is not None
                else None
            )
            stats = _ks_and_channel_fraction(
                region_intervals[interval].to_numpy(dtype=float),
                np_values,
                n_tests,
                alpha,
                channel_alpha,
            )
            rows.append(
                {
                    lobe_col: lobe,
                    region_col: region,
                    "is_lobe_row": False,
                    "interval": interval,
                    "interval_left": left,
                    "interval_right": right,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def test_intervals(
    region_intervals: pd.DataFrame,
    no_peak_intervals: pd.DataFrame,
    print_debug: bool = False,
) -> list[list[tuple[float, float]]]:
    """One-sided KS test (less) per interval vs the no-peak set.

    Returns a list of [(x1, y), (x2, y)] line segments for each significant
    interval, ready to feed to a LineCollection. p-values are Bonferroni
    corrected by 22*42 — to be replaced with Dunnett.
    """
    list_of_intervals = []
    for (idxCol, v1), (_, v2) in zip(region_intervals.items(), no_peak_intervals.items()):
        if print_debug:
            print(v1, v2, idxCol)
        res = ks_2samp(np.asarray(v1), np.asarray(v2), alternative="less", method="asymp")
        t = res.statistic
        pval = res.pvalue * 22 * 42
        if pval < 0.05:
            y = 0.08
            pattern = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")
            bounds = [float(v) for v in pattern.findall(idxCol)]
            list_of_intervals.append([(bounds[0], y), (bounds[1], y)])
            if print_debug:
                print(idxCol, t, format(pval, ".5f"))
    return list_of_intervals


def test_lobes(
    no_peak_df: pd.DataFrame | None,
    psd: pd.DataFrame,
    colbin: pd.Categorical,
) -> dict[str, list[list[tuple[float, float]]]]:
    """Run test_intervals for each lobe vs the no-peak set.

    Lobes are derived from ``psd["Lobe"]``. Returns empty intervals for
    every lobe when no_peak_df is None.
    """
    lobes = list(psd["Lobe"].dropna().unique())
    if no_peak_df is None:
        return {lobe: [] for lobe in lobes}

    freq_cols = _frequency_columns(psd, colbin)
    psd_intervals = get_intervals(psd[freq_cols], colbin)
    psd_intervals = psd_intervals.assign(Lobe=psd["Lobe"].to_numpy())
    no_peak_freq_cols = _frequency_columns(no_peak_df, colbin)
    no_peak_intervals = get_intervals(no_peak_df[no_peak_freq_cols], colbin)

    result = {}
    for lobe in lobes:
        lobe_intervals = psd_intervals[psd_intervals.Lobe == lobe].drop(["Lobe"], axis=1)
        result[lobe] = test_intervals(lobe_intervals, no_peak_intervals)
    return result


def test_regions(
    no_peak_df: pd.DataFrame | None,
    psd: pd.DataFrame,
    colbin: pd.Categorical,
) -> dict[str, dict[str, list[list[tuple[float, float]]]]]:
    """Run test_intervals for each region within each lobe vs the no-peak set.

    Returns nested dict: {lobe: {region: significant_intervals}}.
    Returns empty intervals for every region when no_peak_df is None.
    """
    if no_peak_df is None:
        result = {}
        for lobe in psd["Lobe"].unique():
            regions = psd[psd.Lobe == lobe]["Region name"].unique()
            result[lobe] = {region: [] for region in regions}
        return result

    no_peak_intervals = get_intervals(no_peak_df, colbin)

    result = {}
    for lobe in psd["Lobe"].unique():
        result[lobe] = {}
        regions = psd[psd.Lobe == lobe]["Region name"].unique()
        for region in regions:
            region_psd = psd[(psd.Lobe == lobe) & (psd["Region name"] == region)]
            region_intervals = get_intervals(
                region_psd.drop(["Region name", "Lobe"], axis=1), colbin
            )
            result[lobe][region] = test_intervals(region_intervals, no_peak_intervals)
    return result
