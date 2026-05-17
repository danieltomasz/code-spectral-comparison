from __future__ import annotations

import pandas as pd


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
