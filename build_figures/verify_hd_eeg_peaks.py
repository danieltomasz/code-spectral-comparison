"""Verify the source HD-EEG specparam fit and peak extraction in isolation.

Self-contained copy of the HD-EEG branch of
``5_peak_prevalence_modelled_power.qmd``: load the normalized HD-EEG PSD table,
fit specparam with the BIC-selected **fixed** aperiodic model, extract per-peak
parameters and per-channel fit quality, and print a summary.

Run:
    .venv/bin/python build_figures/verify_hd_eeg_peaks.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from specparam import SpectralGroupModel

PROJECT_DIR = Path("/Users/daniel/PhD/spectral-comparison/code/")
specparam_dir = PROJECT_DIR / "data/interim/specparam_model_selection"
# Same normalized-PSD input the figure doc fits (one row per virtual sensor).
HD_PSD_CSV = specparam_dir / "specparam_input_normalized_psd_source_hd_eeg.csv"

# Identical to the figure doc: settings, fit range, and the BIC-selected mode.
FREQ_RANGE = (1.0, 80.0)
APERIODIC_MODE = "fixed"  # HD-EEG choice from the knee-vs-fixed BIC comparison
SPECPARAM_SETTINGS = dict(
    peak_width_limits=[1, 8],
    max_n_peaks=8,
    min_peak_height=0.1,
    peak_threshold=3.0,
)


def load_psd_csv(path):
    """Split the saved normalized-PSD table into (psd_matrix, freqs, metadata)."""
    df = pd.read_csv(path, index_col=0)
    num_cols = [c for c in df.columns if str(c).replace(".", "", 1).isdigit()]
    freqs = np.array([float(c) for c in num_cols])
    psd = df[num_cols].to_numpy(dtype=float)
    meta = df[[c for c in df.columns if c not in num_cols]].copy()
    meta.index = df.index
    return psd, freqs, meta


def fit_hd_eeg(path, mode=APERIODIC_MODE):
    """Fit every HD-EEG spectrum with the fixed aperiodic model."""
    psd, freqs, meta = load_psd_csv(path)
    fg = SpectralGroupModel(**SPECPARAM_SETTINGS, aperiodic_mode=mode, verbose=False)
    fg.fit(freqs, psd, freq_range=list(FREQ_RANGE), n_jobs=1)
    return fg, meta


def extract_peaks(fg, meta):
    """Tidy per-peak table: channel, region, Lobe, CF, PW, BW."""
    peaks = np.atleast_2d(fg.get_params("peak"))  # (n_peaks, 4): [CF, PW, BW, ID]
    peak_df = pd.DataFrame(peaks, columns=["CF", "PW", "BW", "ID"])
    peak_df["ID"] = peak_df["ID"].astype(int)
    peak_df["channel"] = meta.index.to_numpy()[peak_df["ID"].to_numpy()]
    peak_df["region"] = (
        meta["Region name"].astype(str).str.strip("'").to_numpy()[peak_df["ID"]]
    )
    peak_df["Lobe"] = meta["Lobe"].astype(str).to_numpy()[peak_df["ID"]]
    return peak_df


def channel_fit_quality(fg, channel_names):
    """Per-channel R² (specparam) and MSE of the full log-power model."""
    freqs = np.asarray(fg.data.freqs, dtype=float)
    rows = []
    for i, (channel, result) in enumerate(zip(channel_names, fg.results)):
        ap = fg.modes.aperiodic.func(freqs, *np.asarray(result.aperiodic_fit, dtype=float))
        peak_params = np.asarray(result.peak_fit, dtype=float).ravel()
        peak = fg.modes.periodic.func(freqs, *peak_params) if peak_params.size else 0.0
        residual = np.asarray(fg.data.power_spectra[i], dtype=float) - (ap + peak)
        rows.append(
            {
                "channel": channel,
                "gof_rsquared": result.metrics.get("gof_rsquared", np.nan),
                "mse_log10": float(np.mean(residual**2)),
                "n_peaks": int(len(result.peak_fit)),
            }
        )
    return pd.DataFrame(rows)


def main():
    assert HD_PSD_CSV.exists(), f"missing {HD_PSD_CSV}"
    fg, meta = fit_hd_eeg(HD_PSD_CSV)
    peaks = extract_peaks(fg, meta)
    quality = channel_fit_quality(fg, meta.index.astype(str))

    print(f"input          : {HD_PSD_CSV.name}")
    print(f"aperiodic mode : {APERIODIC_MODE}  | freq range {FREQ_RANGE} Hz")
    print(f"channels       : {len(meta)}")
    print(f"total peaks    : {len(peaks)}  | mean peaks/channel "
          f"{quality['n_peaks'].mean():.2f}")
    print(f"median R²      : {quality['gof_rsquared'].median():.3f}")
    print(f"median MSE     : {quality['mse_log10'].median():.4f}")
    passed = (quality["gof_rsquared"] > 0.9) & (quality["mse_log10"] < 0.1)
    print(f"pass R²>0.9 & MSE<0.1 : {passed.sum()}/{len(quality)} "
          f"({passed.mean():.1%})")

    print("\nfirst peaks:")
    print(peaks.head(10).to_string(index=False))

    out = specparam_dir / "verify_hd_eeg_peaks.csv"
    peaks.merge(quality[["channel", "gof_rsquared", "mse_log10"]], on="channel").to_csv(
        out, index=False
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
