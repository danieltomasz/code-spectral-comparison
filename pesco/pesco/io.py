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

def load_ieeg(DATA_PATH, OUT_PATH, print_debug=False):
    """
    Load intracranial data from file and return time series as an mne object and pandas dataset
    DATA_PATH is the location of the  folder containing the data
    OUT_PATH is the path, where to write mne file

    """
    matlabfile = DATA_PATH + "WakefulnessMatlabFile.mat"
    region_dict_name = DATA_PATH + "RegionInformation.csv"

    matdata = sio.loadmat(matlabfile)
    print(matdata.keys()) if print_debug else True

    ch_names = list(matdata["ChannelName"])
    ch_names = [str(x[0][0]) for x in ch_names]
    ch_types = ["ecog"] * len(ch_names)

    # region =  list(matdata['ChannelRegion'])
    # region = [x[0] for x in  region]

    sfreq = matdata["SamplingFrequency"]
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
    raw.save(OUT_PATH + "ieeg_raw.fif", overwrite=True)
    result.to_csv(OUT_PATH + "ieeg_raw.csv")

    return raw, result


def concat_mat(currentfold, A, matlab_df, result):
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
    hd_dataset_path,
    OUT_PATH=[],
    specific=0,
):
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
    if specific:
        for currentfold in sorted(currentDirectory.iterdir()):
            if currentfold == currentDirectory / str(specific):
                A, matlab_df, result = concat_mat(currentfold, A, matlab_df, result)
    else:
        for currentfold in sorted(currentDirectory.iterdir()):
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
        raw.save(OUT_PATH + "sources_raw.fif", overwrite=True)
        result.to_csv(OUT_PATH + "sources_raw.csv")

    return raw, result

