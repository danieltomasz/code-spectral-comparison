"""Canonical specparam analysis configuration — single source of truth.

Every notebook imports these instead of re-declaring them, so the fit settings
cannot silently drift between analyses. The settings are hashed (see
:func:`pesco.store.params_hash`) and stored alongside every cached result, so a
consumer can assert that the results it loads were produced by *these* settings.
"""

from __future__ import annotations

FREQ_RANGE: tuple[float, float] = (1.0, 80.0)

# specparam peak settings (Afnan-style), shared across modalities and modes.
SPECPARAM_SETTINGS: dict = dict(
    peak_width_limits=[1, 8],
    max_n_peaks=8,
    min_peak_height=0.1,
    peak_threshold=3.0,
)

# Aperiodic mode chosen per modality by the knee-vs-fixed BIC comparison.
SELECTED_MODE: dict[str, str] = {"iEEG atlas": "knee", "source HD-EEG": "fixed"}

# Modality labels used everywhere (dataset column values), in display order.
DATASETS: tuple[str, ...] = ("iEEG atlas", "source HD-EEG")
