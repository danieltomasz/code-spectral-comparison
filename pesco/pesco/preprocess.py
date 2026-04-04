# -*- coding: utf-8 -*-
"""
Loading & prepare data and switch to mne format

"""

import numpy as np
import mne
import pandas as pd
import scipy.signal

import matplotlib.pyplot as plt
import scipy.io
import matplotlib
import scipy


# %% filters
def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5) -> tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filtfilt(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def notch_default(x: np.ndarray, cf: float, bw: float, Fs: float, order: int = 3) -> tuple[np.ndarray, list[np.ndarray]]:
    nyq_rate = Fs / 2.0
    f_range = [cf - bw / 2.0, cf + bw / 2.0]
    Wn = (f_range[0] / nyq_rate, f_range[1] / nyq_rate)
    b, a = scipy.signal.butter(order, Wn, "bandstop")
    return scipy.signal.filtfilt(b, a, x), [b, a]


def filter_network(raw: mne.io.BaseRaw, yes: bool = True) -> mne.io.BaseRaw:
    """filter raw data using mne filter, raw in mne object"""
    if yes:
        notches = (50, 60)
        raw.filter(0.5, 80.0, fir_design="firwin")
        raw.notch_filter(notches, phase="zero-double", fir_design="firwin2")
    return raw




def get_psd_interval(raw: mne.io.BaseRaw, name: str = "power_ieeg.csv", tmin: float = 0.0, tmax: float = 67.0, save_psd: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """return psd for some time period"""
    Fs = raw.info["sfreq"]
    copy = raw.copy().crop(tmin=tmin, tmax=tmax)
    data, times = copy[:, :]

    #
    f, psd = scipy.signal.welch(
        data, Fs, nperseg=int(2 * Fs), noverlap=int(Fs), detrend=False, scaling="spectrum"
    )

    psd_sum = np.sum(psd, 1)
    psd_norm = psd / psd_sum[:, np.newaxis]
    psd_norm = psd_norm
    psd_norm = pd.DataFrame(psd_norm, index=raw.info["ch_names"])

    psd_norm.columns = f
    psd_norm = psd_norm[:, 1:161]

    if save_psd:
        psd_norm.to_csv("../data/interim/" + name)
        np.savetxt("../data/interim/frequency_bins.csv", f, delimiter=",")
    return (psd_norm, f)



def get_psd_mat(data: np.ndarray, Fs: float, ch_names: list[str], name: str = "power_ieeg.csv", save_psd: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """return psd for some time period"""

    # data, times = raw[:,:]
    # data_filt, notch_filtba = notch_default(data, 50, 2, Fs, 3)
    # data_filt, notch_filtba = notch_default(data_filt, 60, 2, Fs, 3)

    #    The spectral density in each channel was estimated with
    #    Welch's method, i.e. averaging the magnitude of the discrete
    #    time Fourier transform of 59 overlapping blocks of 2 s dur-
    #    ation and 1 s step, weighted by a Hamming window. In each
    #    channel the resulting spectral density was normalized to a total
    #    power equal to one, making it independent of the EEG signal
    #    amplitude.
    #
    #
    #    f, psd = scipy.signal.welch(data, Fs, nperseg=2*Fs,noverlap=Fs)
    f, psd = scipy.signal.welch(
        data, Fs, nperseg=int(2 * Fs), noverlap=int(Fs), detrend=False, scaling="spectrum"
    )
    # psd_norm=psd

    # get samples from 0.5 to 80 Hz
    f = f[1:161]
    psd = psd[:, 1:161]
    # get total energy
    total_energy = np.sum(np.sqrt(psd), 1)

    psd_norm = np.sqrt(psd) / total_energy[:, np.newaxis]

    psd_norm = pd.DataFrame(psd_norm, ch_names)
    psd_norm.columns = f

    if save_psd:
        psd_norm.to_csv("../data/interim/" + name)
        np.savetxt("../data/interim/frequency_bins.csv", f, delimiter=",")
    return (f, psd_norm)


def prepare_psd(matfile: str, region_dict_file: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load matlab file  provided with Frauscher, B. et all (2018). Atlas of the normal intracranial electroencephalogram:neurophysiological awake activity in different cortical areas. Brain, 141(4), 1130-1144. https://doi.org/10/gc5ct7
    A pandas dataframe with psd returned
    """

    # df_ieeg = pd.read_csv(datafolder + "ieeg_raw.csv")
    # raw_ieeg = mne.io.read_raw_fif(datafolder +"ieeg_raw.fif",preload=True)

    mat = scipy.io.loadmat(matfile)

    # utils.print_mat_nested(mat, indent=1, nkeys=30)
    patient = mat["Patient"]
    ChannelName = [l.flatten()[0] for l in mat["ChannelName"].flatten()]
    # filtered_list = [ChannelName[i] for i in range(len(ChannelName)) if (patient==1)[i]]
    # specific_chans = raw_ieeg.copy().pick_channels(filtered_list)

    df = pd.DataFrame(patient, columns=["Patient"])
    df["ChannelName"] = ChannelName
    data = mat["Data"].T
    Fs = 200.0

    f, psd_df = get_psd_mat(
        data, Fs, ChannelName, name="power_ieeg.csv", save_psd=False
    )
    psd_df["ChannelRegion"] = mat["ChannelRegion"]
    region_dict = pd.read_csv(region_dict_file)
    psd_df = psd_df.set_index("ChannelRegion").join(region_dict.set_index("Region"))
    return (f, psd_df)



