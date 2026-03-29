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
from pesco.stats import regional_permtest  # noqa: F401
from pesco.peaks import peak_survival  # noqa: F401
