# -*- coding: utf-8 -*-
"""
Loading data and switch to mne format

"""

# %% loading data to mne
import mne
import scipy.io as sio
import pandas as pd
import h5py
import pathlib
import numpy as np
import scipy.io
import scipy.signal

from pesco.preprocess import compute_psd, normalize_psd


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return start/stop pairs for contiguous True runs."""
    runs = []
    start = None
    for sample, value in enumerate(mask):
        if value and start is None:
            start = sample
        if (not value or sample == mask.size - 1) and start is not None:
            stop = sample + int(value and sample == mask.size - 1)
            runs.append((start, stop))
            start = None
    return runs


def _remove_frauscher_zero_buffers(
    data: np.ndarray,
    patients: np.ndarray,
    fs: float,
    duration: float = 60.0,
    min_zero_run: float = 2.0,
    bandpass: tuple[float, float] | None = None,
    filter_order: int = 4,
) -> np.ndarray:
    """Remove atlas zero buffers/padding and return 60 s per channel."""
    expected_samples = int(round(duration * fs))
    min_zero_samples = int(round(min_zero_run * fs))
    patients = np.asarray(patients).ravel()
    filter_coeffs = None
    if bandpass is not None:
        filter_coeffs = scipy.signal.butter(
            filter_order, bandpass, btype="bandpass", fs=fs
        )

    if data.ndim != 2:
        raise ValueError("data must have shape (n_channels, n_samples)")
    if patients.shape[0] != data.shape[0]:
        raise ValueError("patients must have one entry per channel")

    compact = np.empty((data.shape[0], expected_samples), dtype=data.dtype)
    for patient in np.unique(patients):
        channel_idx = np.flatnonzero(patients == patient)
        zero_samples = np.all(data[channel_idx] == 0, axis=0)
        drop = np.zeros_like(zero_samples, dtype=bool)

        for start, stop in _true_runs(zero_samples):
            if stop - start >= min_zero_samples:
                drop[start:stop] = True

        keep = ~drop
        if keep.sum() != expected_samples:
            raise ValueError(
                f"Patient {patient} has {keep.sum()} non-buffer samples; "
                f"expected {expected_samples}."
            )

        if filter_coeffs is None:
            compact[channel_idx] = data[channel_idx][:, keep]
            continue

        b, a = filter_coeffs
        filtered_segments = [
            scipy.signal.filtfilt(b, a, data[channel_idx, start:stop], axis=-1)
            for start, stop in _true_runs(keep)
        ]
        compact[channel_idx] = np.concatenate(filtered_segments, axis=-1)

    return compact


def load_ieeg(DATA_PATH: pathlib.Path, OUT_PATH: pathlib.Path, print_debug: bool = False) -> tuple[mne.io.RawArray, pd.DataFrame]:
    """
    Load intracranial data from file and return time series as an mne object and pandas dataset
    DATA_PATH is the location of the  folder containing the data
    OUT_PATH is the path, where to write mne file

    """
    matlabfile = DATA_PATH / "WakefulnessMatlabFile.mat"
    region_dict_name = DATA_PATH / "RegionInformation.csv"

    matdata = sio.loadmat(matlabfile)
    print(matdata.keys()) if print_debug else True

    ch_names = list(matdata["ChannelName"])
    ch_names = [str(x[0][0]) for x in ch_names]
    ch_types = ["ecog"] * len(ch_names)

    sfreq = matdata["SamplingFrequency"].item()
    data = matdata["Data"].T

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    region = list(matdata["ChannelRegion"])
    region = [x[0] for x in region]

    df = pd.DataFrame({
        "ch_name": ch_names,
        "region": region,
        "region_number": region,
    })

    region_dict = pd.read_csv(region_dict_name)
    result = df.set_index("region").join(region_dict.set_index("Region"))
    dataset = ["ieeg"] * len(result)
    result["dataset"] = dataset
    raw.save(OUT_PATH / "ieeg_raw.fif", overwrite=True)
    result.to_csv(OUT_PATH / "ieeg_raw.csv")

    return raw, result

def prepare_psd(
    matfile: pathlib.Path | str,
    region_dict_file: pathlib.Path | str,
    normalize: bool = True,
    freq_range: tuple[float, float] = (0.5, 80.0),
    remove_zero_buffers: bool = True,
    frauscher_bandpass: bool = True,
    filter_order: int = 4,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load the Frauscher 2018 .mat file and compute per-channel PSD.
 
    Frauscher, B. et al. (2018). Atlas of the normal intracranial EEG.
    Brain, 141(4), 1130-1144. https://doi.org/10/gc5ct7
 
    Parameters
    ----------
    matfile, region_dict_file : path
        Paths to WakefulnessMatlabFile.mat and RegionInformation.csv.
    normalize : bool, default True
        If True, L1-normalize each channel's PSD so its in-band power
        sums to 1 (paper convention; required for the Frauscher
        clustering pipeline). Set False to keep raw V**2/Hz density,
        e.g. for specparam fitting.
    freq_range : (fmin, fmax), default (0.5, 80.0)
        Frequency band kept in the returned PSD. Narrowing this (e.g.
        (1.0, 80.0) for HD-EEG bandpassed at 1 Hz) excludes degenerate
        bins from clustering, peak testing and plotting downstream.
    remove_zero_buffers : bool, default True
        Drop the 2 s zero buffers between artifact-free segments and the
        trailing zero padding in the downloadable atlas file before Welch.
        This reconstructs the 60 s signal used for Frauscher's 59
        overlapping 2 s Welch blocks.
    frauscher_bandpass : bool, default True
        Apply the paper's 0.5-80 Hz band-pass before Welch. When
        remove_zero_buffers is True, filtering is applied separately to the
        valid signal segments so filter transients do not cross zero buffers.
    filter_order : int, default 4
        Butterworth filter order used for the Frauscher band-pass.
 
    Returns
    -------
    f : ndarray
        Frequency bins (Hz) within freq_range.
    psd_df : DataFrame
        Indexed by ChannelRegion (joined to region info). Frequency
        columns followed by 'Region name' and 'Lobe' from
        region_dict_file.
    """
    mat = scipy.io.loadmat(matfile)
    channel_names = [label.flatten()[0] for label in mat["ChannelName"].flatten()]
    data = mat["Data"].T  # (n_channels, n_samples)
    fs = 200.0  # Frauscher data is downsampled to 200 Hz
    if remove_zero_buffers:
        bandpass = (0.5, 80.0) if frauscher_bandpass else None
        data = _remove_frauscher_zero_buffers(
            data,
            mat["Patient"],
            fs,
            bandpass=bandpass,
            filter_order=filter_order,
        )
    elif frauscher_bandpass:
        b, a = scipy.signal.butter(
            filter_order, (0.5, 80.0), btype="bandpass", fs=fs
        )
        data = scipy.signal.filtfilt(b, a, data, axis=-1)
 
    fmin, fmax = freq_range
    f, psd = compute_psd(data, fs, fmin=fmin, fmax=fmax)
    if normalize:
        psd = normalize_psd(psd, f)
 
    psd_df = pd.DataFrame(psd, index=channel_names, columns=f)
    psd_df["ChannelRegion"] = mat["ChannelRegion"]
    region_dict = pd.read_csv(region_dict_file)
    psd_df = psd_df.set_index("ChannelRegion").join(region_dict.set_index("Region"))
    return f, psd_df
 
def concat_mat(
    currentfold: pathlib.Path,
    A: pd.DataFrame,
    matlab_df: pd.DataFrame,
    result: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if currentfold.is_dir():
        filename = str(currentfold.resolve()) + "/roi_data.mat"
        with h5py.File(filename, "r") as f:
            dataset = f["brain_roi"]
            dat = np.array(dataset)
            names = [
                str("dataset" + format(int(currentfold.name[7:]), "02d") + "_")
                + format(i, "02d")
                for i in range(1, 39)
            ]
            names = [val + a for val in names for a in ("R", "L")]
            df = pd.DataFrame(data=dat, columns=names)
            # df= df.set_index(name)
            tempB = pd.concat([A, pd.Series(names)], axis=1)
            result = pd.concat([result, tempB], axis=0)
            matlab_df = pd.concat([matlab_df, df], axis=1)
    return (A, matlab_df, result)


def load_sources(
    hd_dataset_path: pathlib.Path,
    OUT_PATH: pathlib.Path | str = "",
    specific: int = 0,
) -> tuple[mne.io.RawArray, pd.DataFrame]:
    """
    Load intracranial data from file and return time series as an mne object and pandas dataset
    DATA_PATH is the location of the  folder containing the data
    OUT_PATH is the path, where to write mne file
    """

    currentDirectory = pathlib.Path(hd_dataset_path)

    # folders = sorted(os.listdir(DATA_PATH))
    # A is temporaty df with names of regions
    A = pd.read_csv(currentDirectory / "RegionInformation.csv")
    A = pd.DataFrame(A.values.repeat(2, axis=0), columns=A.columns)
    matlab_df = pd.DataFrame()
    result = pd.DataFrame()
    # iter over folders to read the data or load single
    for currentfold in sorted(currentDirectory.iterdir()):
        if specific and int(currentfold.name[7:]) != specific:
            continue
        A, matlab_df, result = concat_mat(currentfold, A, matlab_df, result)
    ch_names = list(matlab_df.columns)
    ch_types = ["eeg"] * len(ch_names)
    sfreq = 200.0
    data = np.nan_to_num(matlab_df.values)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data.T, info)

    result.columns = ["region_number", "Region name", "Lobe", "ch_name"]
    result = result[["ch_name", "region_number", "Region name", "Lobe"]]

    dataset = ["sources"] * len(result)
    result["dataset"] = dataset
    # change to set index proper way
    result = result.set_index("ch_name")
    if OUT_PATH:
        raw.save(pathlib.Path(OUT_PATH) / "sources_raw.fif", overwrite=True)
        result.to_csv(pathlib.Path(OUT_PATH) / "sources_raw.csv")

    return raw, result
