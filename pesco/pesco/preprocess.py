# -*- coding: utf-8 -*-
"""
Loading & prepare data and switch to mne format

"""

import scipy.io as sio
import numpy as np
import mne
import pandas as pd
import scipy.signal
import h5py
import pathlib
import matplotlib.pyplot as plt
import scipy.io
import matplotlib
import scipy


# %% filters
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filtfilt(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def notch_default(x, cf, bw, Fs, order=3):
    nyq_rate = Fs / 2.0
    f_range = [cf - bw / 2.0, cf + bw / 2.0]
    Wn = (f_range[0] / nyq_rate, f_range[1] / nyq_rate)
    b, a = scipy.signal.butter(order, Wn, "bandstop")
    return scipy.signal.filtfilt(b, a, x), [b, a]


def filter_network(raw, yes=True):
    """filter raw data using mne filter, raw in mne object"""
    if yes:
        notches = (50, 60)
        raw.filter(0.5, 80.0, fir_design="firwin")
        raw.notch_filter(notches, phase="zero-double", fir_design="firwin2")
    return raw


# %% loading data to mne


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
    A = pd.read_csv(hd_dataset_path + "/RegionInformation.csv")
    A = pd.DataFrame(A.values.repeat(2, axis=0), columns=A.columns)
    matlab_df = pd.DataFrame()
    result = pd.DataFrame()
    # iter over folders to read the data or load single
    if specific:
        for currentfold in sorted(currentDirectory.iterdir()):
            if str(currentfold) == hd_dataset_path + str(specific):
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


def get_psd_interval(raw, name="power_ieeg.csv", tmin=0.0, tmax=67.0, save_psd=False):
    """return psd for some time period"""
    Fs = raw.info["sfreq"]
    copy = raw.copy().crop(tmin=tmin, tmax=tmax)
    data, times = copy[:, :]

    #
    f, psd = scipy.signal.welch(
        data, Fs, nperseg=2 * Fs, noverlap=Fs, detrend=False, scaling="spectrum"
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


def get_psd_mat(data, Fs, ch_names, name="power_ieeg.csv", save_psd=False):
    """return psd for some time period"""

    # data, times = raw[:,:]
    # data_filt, notch_filtba = notch_default(data, 50, 2, Fs, 3)
    # data_filt, notch_filtba = notch_default(data_filt, 60, 2, Fs, 3)

    #    The spectral density in each channel was estimated with
    #    Welch’s method, i.e. averaging the magnitude of the discrete
    #    time Fourier transform of 59 overlapping blocks of 2 s dur-
    #    ation and 1 s step, weighted by a Hamming window. In each
    #    channel the resulting spectral density was normalized to a total
    #    power equal to one, making it independent of the EEG signal
    #    amplitude.
    #
    #
    #    f, psd = scipy.signal.welch(data, Fs, nperseg=2*Fs,noverlap=Fs)
    f, psd = scipy.signal.welch(
        data, Fs, nperseg=2 * Fs, noverlap=Fs, detrend=False, scaling="spectrum"
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


def prepare_psd(matfile, region_dict_file):
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


def eblow(df, n):
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist, pdist
    import numpy as np

    kMeansVar = [KMeans(n_clusters=k).fit(df.values) for k in range(1, n)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df.values, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df.values) ** 2) / df.values.shape[0]
    bss = tss - wcss
    plt.plot(bss)
    plt.show()


def eblow_psd(psd_df, n):
    from sklearn.cluster import KMeans

    psd_df = psd_df[0:160]

    sse = {}
    for k in range(1, n):
        kmeans = KMeans(n_clusters=k, max_iter=100).fit(psd_df)
        # data["clusters"] = kmeans.labels_
        # print(data["clusters"])
        sse[k] = (
            kmeans.inertia_
        )  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()


def plot_max(psd_df, print_debug=False):
    # maxy = np.zeros([HowManyClusters,160]) #HowManyClusters = numbers of clusters
    HowManyClusters = len(psd_df["clusters"].unique())
    for l in range(0, HowManyClusters):
        MeanClusterSpectrum = np.mean(psd_df["clusters" == l])
        MeanClusterSpectrum = MeanClusterSpectrum[:-1]
        logicalmax = np.zeros([1, HowManyClusters])
        for k in range(0, HowManyClusters):
            # Maxima along the spectrum axis
            maxi = np.amax(psd_df["clusters" == k], 0)
            # last column is for  remembering label, we dont need it temporarily
            maxi = maxi[:-1]
            logicalsum = sum(np.less(MeanClusterSpectrum, maxi))
            if logicalsum == 160:
                logicalmax[0, k] = 1
        if np.sum(logicalmax) == HowManyClusters:
            print(l) if print_debug else True
        # print cluster number, if  averege of it is smaller in every bin than maximal values in other clusters


def plot_single_psd(psd_df, f, i):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.semilogx(f, psd_df.iloc[i])
    ax.grid()
    # ax.plot(f, psd_df.iloc[1])
    ax.set_xlim(0.5, 80)
    ax.set_xticks([0.5, 4, 8, 13, 30, 80])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # plt.yscale('log')
    ax.set_xlabel("Frequency")
    ax.set_ylabel("PSD ")


def compute_clusters(psd_df, HowManyClusters, random_seed=2):
    power = psd_df.values

    from sklearn.preprocessing import Normalizer
    from sklearn.cluster import KMeans
    from sklearn.pipeline import make_pipeline

    normalizer = Normalizer()
    np.random.seed(42)
    np.random.RandomState(3)

    np.random.seed(3)
    kmeans = KMeans(
        n_clusters=HowManyClusters, max_iter=300, n_init=100, random_state=random_seed
    )
    pipeline = make_pipeline(normalizer, kmeans)
    pipeline.fit(power)
    # cluster labels
    cluster_labels = kmeans.labels_
    psd_df = psd_df.assign(clusters=cluster_labels)
    return psd_df


def plot_psd_clusters(psd_df, f, dataset, smal=[], nopeak=[]):
    #    import os
    #    dir = os.path.dirname(__file__)
    #    images = os.path.join(dir, '/images/')
    #    if not os.path.exists(images):
    #       os.makedirs(images)

    psd_medianas = psd_df.groupby("clusters").median()
    # HowManyClusters = len(psd_df["clusters"].unique())
    # plt.close()
    matplotlib.rcParams.update({"font.size": 16})

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # plot means of the clusters
    for k in range(0, len(psd_medianas.index)):
        # srednia = np.mean(psd_df[cluster_labels==k])
        # srednia = srednia[:-1]
        mediana = psd_medianas.loc[k]

        if k in smal and k == nopeak:
            temp_label = (
                "cl. "
                + str(k)
                + " ("
                + str(psd_df["clusters"].value_counts().loc[k])
                + " el.)"
            )
            ax.semilogx(f, mediana, linewidth=4.0, color="black", label=temp_label)
        elif k == nopeak:
            temp_label = (
                "cl. "
                + str(k)
                + " ("
                + str(psd_df["clusters"].value_counts().loc[k])
                + " el.)"
            )
            ax.semilogx(f, mediana, linestyle=":", linewidth=5.0, label=temp_label)

        else:
            temp_label = (
                "cl. "
                + str(k)
                + " ("
                + str(psd_df["clusters"].value_counts().loc[k])
                + " el.)"
            )

            ax.semilogx(f, mediana, alpha=0.5, label=temp_label)

        # plt.xscale('log')
        # plt.xlim(0.5, 80)
        # xticks = [1, 2, 4, 8, 16, 32, 64]
        # ticklabels = ['1', '2', '4', '8', '16', '32', '64']
        # plt.xticks(xticks, ticklabels)

    # ax.legend(range(0,len(psd_medianas.index))  )
    ax.legend()
    ax.set_xticks([0.5, 4, 8, 13, 30, 80])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid()
    coordinates = [2, 6, 10, 16, 36]

    textes = [
        r"""$\delta$""",
        r"""$\theta$""",
        r"""$\alpha$""",
        r"""$\beta$""",
        r"""$\gamma$""",
    ]
    for t, text in zip(coordinates, textes):
        ax.text(t, 0.05, text, fontsize=14)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Normalized spectral density")
    plt.title("Median power of different  PSDs clusters of " + dataset)
    plt.savefig("images/" + dataset + "_clusters.svg", format="svg")
    plt.show()
    return (fig, ax)


def plot_specific_clusterisation(
    psd, f, HowManyClusters, seed, dataset, ifall=False, nopeak=[], print_debug=False
):
    psd_df = compute_clusters(psd, HowManyClusters, seed)
    psd_medianas = psd_df.groupby("clusters").median()
    smal = []
    print() if print_debug else True
    for index, row in psd_medianas.iterrows():
        #            temp = psd_medianas.copy.drop(psd_medianas.index[index])

        max_completion = psd_medianas.iloc[psd_medianas.index != index, :].max()
        smaller = np.sum(np.less(row, max_completion))
        # print( "cluster ", index , " is smaller on  ", smaller , "frequencies")
        if smaller == 160:
            print() if print_debug else True
            print(
                "cluster", index, "when we have", HowManyClusters, "clusters"
            ) if print_debug else True
            smal.append(index)
    if ifall:
        for nopeak in range(0, HowManyClusters):
            fig, ax = plot_psd_clusters(psd_df, f, dataset, smal, nopeak)
    else:
        fig, ax = plot_psd_clusters(psd_df, f, dataset, smal, nopeak)
        print(nopeak) if print_debug else True
    print(psd_df["clusters"].value_counts()) if print_debug else True
    return psd_df, smal
