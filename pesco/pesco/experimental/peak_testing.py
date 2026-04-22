"""Stage 3 of the Frauscher 2018 pipeline: bin into frequency intervals and KS-test against no-peak."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# Paper Table: 22 frequency intervals (Hz boundaries)
_PAPER_INTERVALS = (
    0.5, 0.75, 1.25, 1.75, 2.25, 3.25, 3.75, 4.25, 5.25, 6.25, 6.75,
    7.75, 8.25, 9.25, 10.25, 11.75, 13.25, 15.25, 17.25, 20.25, 24.25,
    31.75, 80,
)


def cutintervals(x: np.ndarray) -> tuple[pd.Categorical, np.ndarray]:
    """Bin frequencies into the paper's 22 intervals."""
    colBin, y = pd.cut(x, _PAPER_INTERVALS, retbins=True, include_lowest=True)
    return colBin, y


def get_intervals(psd: pd.DataFrame, colbin: pd.Categorical) -> pd.DataFrame:
    """Sum PSD power within each frequency interval."""
    dictionary = dict(zip(psd.columns, colbin))
    psd_intervals = psd.T.groupby(dictionary).sum().T
    psd_intervals.columns = psd_intervals.columns.astype(str, copy=False)
    return psd_intervals


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
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector

    rstats = importr("stats")
    list_of_intervals = []
    for (idxCol, v1), (_, v2) in zip(region_intervals.items(), no_peak_intervals.items()):
        if print_debug:
            print(v1, v2, idxCol)
        v1 = FloatVector(v1)
        v2 = FloatVector(v2)
        htest = rstats.ks_test(v1, v2, alternative="less", exact=False)
        htestlist = list(htest)
        t = htestlist[0][0]
        pval = htestlist[1][0]
        pval = pval * 22 * 42
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

    Returns empty intervals for every lobe when no_peak_df is None.
    """
    lobes = ["Occipital", "Parietal", "Frontal", "Temporal"]
    if no_peak_df is None:
        return {lobe: [] for lobe in lobes}

    psd_intervals = get_intervals(psd, colbin)
    psd_intervals = psd_intervals.assign(Lobe=psd["Lobe"])
    no_peak_intervals = get_intervals(no_peak_df, colbin)

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