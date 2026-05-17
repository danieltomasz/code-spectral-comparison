import numpy as np
import pandas as pd

from pesco.experimental.peak_testing import (
    cutintervals,
    test_regions_heatmap as run_regions_heatmap,
)


def test_regions_heatmap_uses_corrected_ks_gate_for_channel_fraction():
    f = np.array([1.0, 2.0, 5.0, 6.0])
    colbin, _ = cutintervals(f, edges=np.array([0.5, 3.0, 7.0]))

    no_peak_df = pd.DataFrame(
        np.tile([1.0, 1.0, 1.0, 1.0], (30, 1)),
        columns=f,
    )
    psd = pd.DataFrame(
        {
            1.0: [10.0, 11.0, 12.0, 13.0, 1.0, 1.0, 1.0, 1.0],
            2.0: [10.0, 11.0, 12.0, 13.0, 1.0, 1.0, 1.0, 1.0],
            5.0: [1.0] * 8,
            6.0: [1.0] * 8,
            "Region name": ["High"] * 4 + ["Flat"] * 4,
            "Lobe": ["Frontal"] * 8,
        }
    )

    result = run_regions_heatmap(no_peak_df, psd, colbin)

    # 2 regions x 2 intervals + 1 lobe x 2 intervals
    assert len(result) == 6
    assert result["channel_fraction"].dropna().between(0, 1).all()
    assert result.loc[result["is_lobe_row"], "Region name"].eq("Frontal").all()

    region_rows = result[~result["is_lobe_row"]]
    high_low_freq = region_rows[
        (region_rows["Region name"] == "High")
        & (region_rows["interval_left"] < 3.0)
    ].iloc[0]
    assert high_low_freq["ks_significant"]
    assert high_low_freq["channel_fraction"] == 1.0
    assert high_low_freq["n_channel_significant"] == 4

    flat_low_freq = region_rows[
        (region_rows["Region name"] == "Flat")
        & (region_rows["interval_left"] < 3.0)
    ].iloc[0]
    assert not flat_low_freq["ks_significant"]
    assert np.isnan(flat_low_freq["channel_fraction"])

    result_no_lobes = run_regions_heatmap(
        no_peak_df, psd, colbin, include_lobes=False
    )
    assert len(result_no_lobes) == 4
    assert not result_no_lobes["is_lobe_row"].any()
