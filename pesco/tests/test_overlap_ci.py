import matplotlib

import numpy as np
import pandas as pd

matplotlib.use("Agg")

from pesco.spectral import afnan_band_overlap, bootstrap_overlap_ci  # noqa: E402
from pesco.experimental.plotting import plot_overlap_frauscher_heatmap  # noqa: E402

FREQS = np.arange(1.0, 81.0)
REGIONS = ["A reg", "B reg", "C reg"]


def _toy(subjects, seed):
    """Channels x (freq cols + Region name); 3 channels per (subject, region)."""
    rng = np.random.default_rng(seed)
    rows, rcol, subj = [], [], []
    for s in subjects:
        for r in REGIONS:
            for _ in range(3):
                rows.append(1.0 / FREQS + rng.normal(0, 0.05, FREQS.size))
                rcol.append(r)
                subj.append(s)
    df = pd.DataFrame(rows, columns=FREQS)
    df["Region name"] = rcol
    return df, np.array(subj)


def test_bootstrap_overlap_ci_shapes_bounds_and_determinism():
    ref_df, ref_s = _toy(["s1", "s2", "s3"], seed=0)
    est_df, est_s = _toy(["d1", "d2", "d3"], seed=1)

    point, lo, hi = bootstrap_overlap_ci(
        ref_df, est_df, FREQS, FREQS, ref_s, est_s, B=100, seed=3
    )

    # point matches the un-resampled metric, and lo/hi share its grid.
    expected = afnan_band_overlap(ref_df, est_df, FREQS, FREQS)
    pd.testing.assert_frame_equal(point, expected)
    assert lo.shape == hi.shape == point.shape

    finite = np.isfinite(lo.to_numpy()) & np.isfinite(hi.to_numpy())
    assert (lo.to_numpy()[finite] <= hi.to_numpy()[finite] + 1e-9).all()

    # same seed -> identical interval (reproducible).
    _, lo2, hi2 = bootstrap_overlap_ci(
        ref_df, est_df, FREQS, FREQS, ref_s, est_s, B=100, seed=3
    )
    pd.testing.assert_frame_equal(lo, lo2)
    pd.testing.assert_frame_equal(hi, hi2)


def test_plot_overlap_frauscher_heatmap_orders_rows_by_lobe():
    ref_df, ref_s = _toy(["s1", "s2"], seed=0)
    est_df, est_s = _toy(["d1", "d2"], seed=1)
    point, lo, _ = bootstrap_overlap_ci(
        ref_df, est_df, FREQS, FREQS, ref_s, est_s, B=50, seed=3
    )
    region_lobe = {"A reg": "Frontal", "B reg": "Parietal", "C reg": "Occipital"}

    fig, ax = plot_overlap_frauscher_heatmap(
        point, region_lobe, grey=(lo <= 0.0), dot=(point > 0.5) & (lo > 0.0)
    )

    # _LOBE_ORDER puts Occipital first, then Parietal, then Frontal.
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["C reg", "B reg", "A reg"]
    assert fig is not None
