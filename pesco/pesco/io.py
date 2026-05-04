# -*- coding: utf-8 -*-
"""
Loading data and switch to mne format

"""

# %% loading data to mne
from __future__ import annotations
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


def _load_frauscher_mat(
    matfile: pathlib.Path | str,
    region_dict_file: pathlib.Path | str,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Load atlas .mat, return (data, fs, meta).

    Parameters
    ----------
    matfile, region_dict_file : path
        Paths to WakefulnessMatlabFile.mat and RegionInformation.csv.

    Returns
    -------
    data : ndarray, shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz.
    meta : DataFrame
        Indexed by channel name. Columns: ChannelRegion, patient,
        mni_x, mni_y, mni_z, Region name, Lobe.
    """
    mat = scipy.io.loadmat(matfile)
    ch_names: list[str] = [str(x[0][0]) for x in mat["ChannelName"]]
    fs = float(mat["SamplingFrequency"].item())
    data: np.ndarray = mat["Data"].T

    region_num: np.ndarray = mat["ChannelRegion"].ravel()
    pos: np.ndarray = mat["ChannelPosition"]
    patient: np.ndarray = mat["Patient"].ravel()

    meta = pd.DataFrame(
        {
            "ChannelRegion": region_num,
            "patient": patient,
            "mni_x": pos[:, 0],
            "mni_y": pos[:, 1],
            "mni_z": pos[:, 2],
        },
        index=pd.Index(ch_names, name="channel"),
    )

    region_dict = pd.read_csv(region_dict_file)
    meta = meta.join(region_dict.set_index("Region"), on="ChannelRegion")
    return data, fs, meta


def load_ieeg(
    data_path: pathlib.Path | str,
    out_path: pathlib.Path | str | None = None,
) -> tuple[mne.io.RawArray, pd.DataFrame]:
    """Atlas → mne.Raw + metadata. Optional disk dump if out_path given."""
    data_path = pathlib.Path(data_path)
    data, fs, meta = _load_frauscher_mat(
        data_path / "WakefulnessMatlabFile.mat",
        data_path / "RegionInformation.csv",
    )
    info = mne.create_info(ch_names=list(meta.index), sfreq=fs, ch_types="ecog")
    raw = mne.io.RawArray(data, info)
    if out_path is not None:
        out_path = pathlib.Path(out_path)
        raw.save(out_path / "ieeg_raw.fif", overwrite=True)
        meta.to_csv(out_path / "ieeg_raw.csv")
    return raw, meta


def prepare_psd(
    matfile: pathlib.Path | str,
    region_dict_file: pathlib.Path | str,
    normalize: bool = True,
    freq_range: tuple[float, float] = (0.5, 80.0),
    remove_zero_buffers: bool = True,
    frauscher_bandpass: bool = True,
    filter_order: int = 4,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load atlas .mat and compute per-channel PSD.

    Returns
    -------
    f : ndarray, shape (n_freqs,)
    psd_df : DataFrame
        Indexed by channel name. Frequency columns (float, Hz) followed
        by metadata: ChannelRegion, patient, mni_x/y/z, Region name, Lobe.
    """
    data, fs, meta = _load_frauscher_mat(matfile, region_dict_file)

    if remove_zero_buffers:
        bandpass: tuple[float, float] | None = (
            (0.5, 80.0) if frauscher_bandpass else None
        )
        data = _remove_frauscher_zero_buffers(
            data,
            meta["patient"].to_numpy(),
            fs,
            bandpass=bandpass,
            filter_order=filter_order,
        )
    elif frauscher_bandpass:
        b, a = scipy.signal.butter(filter_order, (0.5, 80.0), btype="bandpass", fs=fs)
        data = scipy.signal.filtfilt(b, a, data, axis=-1)

    fmin, fmax = freq_range
    f, psd = compute_psd(data, fs, fmin=fmin, fmax=fmax)
    if normalize:
        psd = normalize_psd(psd, f)

    psd_df = pd.DataFrame(psd, index=meta.index, columns=f).join(meta)
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
