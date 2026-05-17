#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:56:46 2018

@author: daniel
"""

from specparam import SpectralGroupModel


def _extract_results(fg):
    """Extract results from SpectralGroupModel (specparam v2 API)."""
    import pandas as pd
    import numpy as np

    temp_df = pd.DataFrame()
    temp_df["sls"] = [r.aperiodic_fit[-1] for r in fg.results]
    temp_df["errors"] = [r.metrics["error_mae"] for r in fg.results]
    temp_df["r2s"] = [r.metrics["gof_rsquared"] for r in fg.results]
    temp_df["ch_number"] = np.arange(len(fg.results))

    rows = []
    for ch_idx, r in enumerate(fg.results):
        for peak in r.peak_fit.reshape(-1, 3):
            rows.append([*peak, ch_idx])
    peaks_df = pd.DataFrame(rows, columns=["CF", "Amp", "BW", "ch_number"])

    return temp_df, peaks_df


# Getting the peaks - Voytek method
def get_peaks(psd, f):
    """return peaks using specparam method"""
    freq_range = [2, 40]
    fg = SpectralGroupModel(
        peak_width_limits=[1, 12],
        min_peak_height=0.2,
        max_n_peaks=6,
        aperiodic_mode="knee",
        peak_threshold=3.0,
    )
    fg.fit(f, psd, freq_range)
    _, peaks_df = _extract_results(fg)
    return peaks_df


def comp_foof(raw, chan_details, outputfile):
    import pandas as pd
    import numpy as np
    from neurodsp.spectral import compute_spectrum

    data, times = raw[:]
    Fs = raw.info["sfreq"]
    freqs, spectrum = compute_spectrum(data, Fs, method="welch", nperseg=int(2 * Fs))
    psd_df = pd.DataFrame(spectrum, index=chan_details["region_number"])
    fg = SpectralGroupModel(peak_width_limits=[1, 8], min_peak_height=0.05, max_n_peaks=6)
    fg.fit(freqs, psd_df.values, freq_range=[3, 40], n_jobs=-1)

    temp_df, peaks_df = _extract_results(fg)
    chan_details["ch_number"] = np.arange(len(chan_details))
    peaks_all = (
        chan_details
        .set_index("ch_number")
        .join(peaks_df.set_index("ch_number"))
        .join(temp_df.set_index("ch_number"))
    )
    peaks_all.to_csv(outputfile)

    return peaks_all


def comp_foof_seg(raw, chan_details, tmin, tmax):
    import pandas as pd
    import numpy as np
    import mne

    fmin, fmax = 3, 80
    srate = raw.info["sfreq"]
    n_fft, n_overlap, n_per_seg = int(2 * srate), int(srate), int(2 * srate)
    spectrum, freqs = mne.time_frequency.compute_raw_psd(
        raw,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        n_fft=n_fft,
        n_overlap=n_overlap,
        n_per_seg=n_per_seg,
        verbose=False,
    )
    psd_df = pd.DataFrame(spectrum, index=chan_details["region_number"])
    fg = SpectralGroupModel(peak_width_limits=[1, 8], min_peak_height=0.05, max_n_peaks=6)
    fg.fit(freqs, psd_df.values, freq_range=[3, 40], n_jobs=-1)

    temp_df, peaks_df = _extract_results(fg)
    chan_details["ch_number"] = np.arange(len(chan_details))
    peaks_all = (
        chan_details
        .set_index("ch_number")
        .join(peaks_df.set_index("ch_number"))
        .join(temp_df.set_index("ch_number"))
    )
    return peaks_all
