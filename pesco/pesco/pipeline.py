"""Per-dataset namespace for the Frauscher-style pipeline.

Holds intermediate state across stages so multiple datasets can run in the
same notebook without name collisions. Fields are populated by direct
assignment from the existing stage functions; nothing here wraps them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DatasetCtx:
    """Bundle of pipeline state for one dataset.

    Populate fields by direct assignment as you run stages. Field defaults
    are ``None`` so a forgotten stage produces a clear error at the next
    use site rather than silently propagating stale data.
    """
    name: str
    f: np.ndarray = field(default_factory=lambda: np.array([]))
    psd: pd.DataFrame = field(default_factory=pd.DataFrame)
    psd_clust: pd.DataFrame = field(default_factory=pd.DataFrame)
    smal: list[int] = field(default_factory=list)
    no_peak_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    no_peak_center: np.ndarray = field(default_factory=lambda: np.array([]))
    sig_lobes: dict = field(default_factory=dict)
    sig_regions: dict = field(default_factory=dict)
    regional_diff: pd.DataFrame = field(default_factory=pd.DataFrame)
    peaks: dict = field(default_factory=dict)
    # colbin awkward — keep Optional or skip
    colbin: pd.Categorical | None = None