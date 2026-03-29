#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:56:46 2018

@author: daniel
"""

from fooof import FOOOFGroup


# Getting the peaks - Voytek method
def get_peaks(psd, f):
    """return peaks using Voytek foof method"""
    freq_range = [2, 40]
    fg = FOOOFGroup(
        peak_width_limits=[1, 12],
        min_peak_amplitude=0.2,
        max_n_peaks=6,
        background_mode="knee",
        peak_threshold=3.0,
    )
    fg.fit(f, psd, freq_range)
    return fg.get_all_data("peak_params")


def comp_foof(raw, chan_details, outputfile):
    import pandas as pd
    import numpy as np
    from neurodsp.spectral import compute_spectrum
    from fooof import FOOOFGroup

    data, times = raw[:]
    Fs = raw.info["sfreq"]
    freqs, spectrum = compute_spectrum(data, Fs, method="median", nperseg=2 * Fs)
    psd_df = pd.DataFrame(spectrum, index=chan_details["region_number"])
    fg = FOOOFGroup(peak_width_limits=[1, 8], min_peak_amplitude=0.05, max_n_peaks=6)
    fg.fit(freqs, psd_df.values, freq_range=[3, 40], n_jobs=-1)

    # extracting data from  fg
    temp_df = pd.DataFrame()  # prepare error and slope dataframe
    temp_df["sls"] = fg.get_all_data("background_params", "slope")
    temp_df["errors"] = fg.get_all_data("error")
    temp_df["r2s"] = fg.get_all_data("r_squared")
    temp_df["ch_number"] = np.arange(len(chan_details))

    peaks = fg.get_all_data("peak_params")  # prepare peaks dataframe
    peaks_df = pd.DataFrame(peaks)
    peaks_df.columns = ["CF", "Amp", "BW", "ch_number"]

    chan_details["ch_number"] = np.arange(len(chan_details))  # prepare ch_details
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
    from neurodsp.spectral import compute_spectrum
    from fooof import FOOOFGroup

    import mne

    fmin, fmax = 3, 80
    # PSD settings
    srate = raw.info["sfreq"]
    n_fft, n_overlap, n_per_seg = int(2 * srate), int(srate), int(2 * srate)
    data, times = raw[:]
    Fs = raw.info["sfreq"]
    # freqs, spectrum = compute_spectrum(data, Fs,  method='median', nperseg=2*Fs)
    spectrum, freqs = mne.time_frequency.psd_welch(
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
    fg = FOOOFGroup(peak_width_limits=[1, 8], min_peak_amplitude=0.05, max_n_peaks=6)
    fg.fit(freqs, psd_df.values, freq_range=[3, 40], n_jobs=-1)

    # extracting data from  fg
    temp_df = pd.DataFrame()  # prepare error and slope dataframe
    temp_df["sls"] = fg.get_all_data("background_params", "slope")
    temp_df["errors"] = fg.get_all_data("error")
    temp_df["r2s"] = fg.get_all_data("r_squared")
    temp_df["ch_number"] = np.arange(len(chan_details))

    peaks = fg.get_all_data("peak_params")  # prepare peaks dataframe
    peaks_df = pd.DataFrame(peaks)
    peaks_df.columns = ["CF", "Amp", "BW", "ch_number"]

    chan_details["ch_number"] = np.arange(len(chan_details))  # prepare ch_details
    peaks_all = (
        chan_details
        .set_index("ch_number")
        .join(peaks_df.set_index("ch_number"))
        .join(temp_df.set_index("ch_number"))
    )
    return peaks_all
