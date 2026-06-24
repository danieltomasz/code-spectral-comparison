__version__ = '0.0.1.dev0'

import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from pesco import irasa  # noqa: F401
from pesco.spectral import (  # noqa: F401
    aperiodic_curve,
    specparam2pandas,
    compute_curvature_q,
    inspect_fits,
    inspect_fit_quality,
    inspect_q_extremes,
)
from pesco.bandpower import (  # noqa: F401
    relative_band_power_by_channel,
    region_band_power,
    compare_region_band_power,
    plot_band_power_correlation_grid,
    relative_psd_df,
)
from pesco.stats import regional_permtest  # noqa: F401
from pesco.peaks import (  # noqa: F401
    dataset_peaks,
    no_peak_fraction,
    no_peak_table,
    peak_survival,
    plot_no_peak_fraction,
    plot_peak_centre_frequency,
)
