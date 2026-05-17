# -*- coding: utf-8 -*-
"""
Loading & prepare data and switch to mne format

"""

import numpy as np
import mne
import pandas as pd
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib
import scipy

import numpy as np
import scipy.signal

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
        notches = (50)
        raw.filter(0.5, 80.0, fir_design="firwin")
        raw.notch_filter(notches, phase="zero-double", fir_design="firwin2")
    return raw



def compute_psd(
    data: np.ndarray,
    fs: float,
    fmin: float = 0.5,
    fmax: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD following Frauscher et al. 2018.

    Hamming window, 2 s segments, 1 s step (50% overlap), density scaling.
    Returns unnormalized PSD in V**2/Hz, suitable for specparam.
    For Frauscher-style clustering, apply normalize_psd to the result.

    Parameters
    ----------
    data : array, shape (..., n_samples)
        Time series. PSD is computed along the last axis.
    fs : float
        Sampling frequency in Hz.
    fmin, fmax : float
        Frequency band to return (inclusive).

    Returns
    -------
    f : array, shape (n_freqs,)
    psd : array, shape (..., n_freqs)
    """
    nperseg = int(2 * fs)
    noverlap = int(fs)
    f, psd = scipy.signal.welch(
        data,
        fs=fs,
        window="hamming",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
    )
    band = (f >= fmin) & (f <= fmax)
    return f[band], psd[..., band]


def normalize_psd(psd: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Normalize each channel's PSD so its in-band integral equals 1.

    Paper convention (Frauscher 2018): ``∫ psd df = 1`` over the kept
    band, i.e. a true density. Bin-width invariant — values stay the
    same if ``nperseg`` changes. Do NOT apply before specparam.
    """
    df = f[1] - f[0]
    return psd / (psd.sum(axis=-1, keepdims=True) * df)


def normalize_psd_ratio_to_mean(psd: np.ndarray) -> np.ndarray:
    """Express each PSD as ratio to grand-mean PSD (Keitel & Gross 2016).

    Paper convention: "expressed each single-segment power spectrum for each
    brain area as ratio to the mean power spectrum (averaged across all
    segments and brain areas)". Emphasises regionally specific spectral
    profiles by removing the shared 1/f background.

    Parameters
    ----------
    psd : array, shape (..., n_freqs)
        PSD with frequency on the last axis. Any leading axes (channels,
        segments, ...) are averaged together to form the grand mean.

    Returns
    -------
    array of same shape as ``psd``: ``psd / psd.mean(axis=leading_axes)``.
    """
    if isinstance(psd, pd.DataFrame):
        numeric = psd.select_dtypes(include=np.number)
        arr = numeric.to_numpy()
        grand_mean = arr.mean(axis=tuple(range(arr.ndim - 1)), keepdims=True)
        return pd.DataFrame(arr / grand_mean, index=numeric.index, columns=numeric.columns)
    arr = np.asarray(psd)
    grand_mean = arr.mean(axis=tuple(range(arr.ndim - 1)), keepdims=True)
    return arr / grand_mean