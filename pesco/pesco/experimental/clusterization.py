#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% [markdown]
# # Frequency peaks in intracranial and  reconstructed sources data
#
# %%
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.ticker
import sklearn.metrics.pairwise

from scipy import stats

seed = 3

# for importing r

# %%

def _elbow_scores(df: pd.DataFrame, n: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist, pdist

    kMeansVar = [KMeans(n_clusters=k).fit(df.values) for k in range(1, n)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df.values, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = np.array([sum(d**2) for d in dist])
    tss = sum(pdist(df.values) ** 2) / df.values.shape[0]
    bss = tss - wcss
    return bss


def eblow(df: pd.DataFrame, n: int) -> None:
    bss = _elbow_scores(df, n)
    plt.plot(bss)
    plt.show()


def _elbow_scores_psd(psd_df: pd.DataFrame, n: int) -> dict[int, float]:
    from sklearn.cluster import KMeans

    psd_df = psd_df[0:160]
    sse = {}
    for k in range(1, n):
        kmeans = KMeans(n_clusters=k, max_iter=100).fit(psd_df)
        sse[k] = kmeans.inertia_  # Sum of distances of samples to their closest cluster center
    return sse


def eblow_psd(psd_df: pd.DataFrame, n: int) -> None:
    sse = _elbow_scores_psd(psd_df, n)
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()


def compute_clusters(psd_df: pd.DataFrame, HowManyClusters: int, random_seed: int = 2) -> pd.DataFrame:
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




def identify_small_clusters(psd_medianas: pd.DataFrame) -> list[int]:
    """Return indices of clusters whose median is below all other clusters' max at every frequency."""
    smal = []
    for index, row in psd_medianas.iterrows():
        max_completion = psd_medianas.iloc[psd_medianas.index != index, :].max()
        if np.sum(np.less(row, max_completion)) == 160:
            smal.append(index)
    return smal



def get_no_peak(psd_clust: pd.DataFrame, smal: list[int]) -> tuple[pd.DataFrame, np.ndarray]:
    no_peak = psd_clust[psd_clust["clusters"] == smal[0]]
    no_peak = no_peak[no_peak.columns[:-1]]
    median = np.median(no_peak, 0)
    temp_med = np.tile(median, (len(no_peak), 1))
    no_peak["distances_to_median"] = sklearn.metrics.pairwise.paired_distances(
        no_peak, temp_med
    )
    df = no_peak[
        no_peak.distances_to_median < no_peak.distances_to_median.quantile(0.50)
    ]
    return df, median





# plot by lobe
def cutintervals(x: np.ndarray) -> tuple[pd.Categorical, np.ndarray]:
    intervals = (
        0.5,
        0.75,
        1.25,
        1.75,
        2.25,
        3.25,
        3.75,
        4.25,
        5.25,
        6.25,
        6.75,
        7.75,
        8.25,
        9.25,
        10.25,
        11.75,
        13.25,
        15.25,
        17.25,
        20.25,
        24.25,
        31.75,
        80,
    )
    colBin, y = pd.cut(x, intervals, retbins=True, include_lowest=True)
    return colBin, y


def get_intervals(psd: pd.DataFrame, colbin: pd.Categorical) -> pd.DataFrame:
    dictionary = dict(zip(psd.columns, colbin))
    psd_intervals = psd.T.groupby(dictionary).sum().T
    psd_intervals.columns = psd_intervals.columns.astype(str, copy=False)
    return psd_intervals


def plot_intervals(psd_intervals: pd.DataFrame, i: int, dataset: str) -> None:

    #
    fig, ax = plt.subplots(1, 2, sharex=False, figsize=(30, 15))
    ax = ax.flatten()
    matplotlib.rcParams.update({"font.size": 22})
    bins = np.array([
        0.25,
        0.75,
        1.25,
        1.75,
        2.25,
        3.25,
        3.75,
        4.25,
        5.25,
        6.25,
        6.75,
        7.75,
        8.25,
        9.25,
        10.25,
        11.75,
        13.25,
        15.25,
        17.25,
        20.25,
        24.25,
        31.75,
        80,
    ])
    if "Lobe" in psd_intervals.columns:
        freqs = psd_intervals.iloc[i].drop(["Lobe"]).values
    else:
        freqs = psd_intervals.iloc[i].values
    # freqs = psd_intervals_mean.values
    fig.suptitle(dataset)

    ax[0].fill_between(
        bins.repeat(2)[1:-1], freqs.astype(float).repeat(2), facecolor="steelblue"
    )
    ax[0].hlines(y=0.04, xmin=0.5, xmax=80, linewidth=2, color="r")
    ax[0].set_title(" absolute share of power in a given bin")
    ax[0].set_xscale("log")
    ax[0].set_xlim([0.5, 80])
    ax[0].set_xticks([0.5, 4, 8, 13, 30, 80])
    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0].grid()
    ax[0].set_xticks([0.5, 4, 8, 13, 30, 80])
    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax[0].set_ylim(0,max(freqs.astype(np.float)))
    ax[0].set_ylim(0, 0.40)
    # normalized plot
    widths = bins[1:] - bins[:-1]
    heights = freqs.astype(float) / widths
    ax[1].fill_between(bins.repeat(2)[1:-1], heights.repeat(2), facecolor="steelblue")
    ax[1].set_title("share of power in a given bin normalized by width of the bin ")
    ax[1].set_xscale("log")
    #    ax = plt.gca(); cur_ylim = ax.get_ylim(); ax[1].set_ylim(0,cur_ylim[1])
    ax[1].set_xticks([0.5, 4, 8, 13, 30, 80])
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].grid()
    ax[1].set_xlim([0.5, 80])
    ax[1].set_ylim(0, 0.07)
    # ax[1].set_ylim(0,max(heights.astype(np.float)))
    plt.savefig("images/mean_power_share_in_intervals.png", format="png")
    plt.show()


def plot_ecdf(v1: np.ndarray, v2: np.ndarray, lobe: str, idxCol: str, interval: list[float]) -> None:
    from mlxtend.plotting import ecdf

    fig, ax = plt.subplots()
    ax, _, _ = ecdf(v1)
    # second ecdf
    # x2 = X[:, 1]
    ax, _, _ = ecdf(v2, ax=ax)
    ax.set_xlabel("Normalized spectral density in bin")
    ax.set_ylabel("ECDF")
    ax.set_title(lobe + idxCol)
    ax.legend(["lobe", "no peak"], loc="upper left")
    plt.savefig(
        "images/" + lobe + "_" + str(interval[0]) + "_" + str(interval[1]) + "_interval.svg",
        format="svg",
    )


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def return_signifcant(temp: pd.DataFrame, df_intervals: pd.DataFrame, print_debug: bool = False) -> list[list[tuple[float, float]]]:
    import re
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import FloatVector

    rstats = importr("stats")
    list_of_intervals = []
    for (idxCol, v1), (_, v2) in zip(temp.items(), df_intervals.items()):
        print(v1, v2, idxCol) if print_debug else True
        # v1 is a column from lobes
        # v2 is  a column from no peak set
        t, pval = stats.ks_2samp(v1, v2)
        # details of r function https://rdrr.io/cran/dgof/man/ks.test.html
        # ther is no computation of True value only assymptotic aproximation
        v1 = FloatVector(v1)
        v2 = FloatVector(v2)
        htest = rstats.ks_test(v1, v2, alternative="less", exact=False)
        htestlist = list(htest)
        t = htestlist[0][0]
        pval = htestlist[1][0]
        pval = pval * 22 * 42
        p = 0.05
        y = 0.08
        if pval < p:
            pattern = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")
            bounds = [float(v) for v in pattern.findall(idxCol)]
            list_of_intervals.append([(bounds[0], y), (bounds[1], y)])
            # print('list of intervals', *list_of_intervals, sep = ", ")
            print(idxCol, t, format(pval, ".5f")) if print_debug else True
            # plot_ecdf(v1, v2, lobe, idxCol, l)
            # i = i + 1
    return list_of_intervals


def return_signifcant_lobes(df: pd.DataFrame, psd: pd.DataFrame, colbin: pd.Categorical, print_debug: bool = False) -> dict[str, list[list[tuple[float, float]]]]:

    # psd_intervals = psd_intervals.astype(float)
    psd_intervals = get_intervals(psd, colbin)
    psd_intervals = psd_intervals.assign(Lobe=psd["Lobe"])
    df_intervals = get_intervals(df, colbin)

    stat = pd.DataFrame(index=["stat_value", "pvalue"], columns=psd_intervals.columns)
    pd.set_option("future.no_silent_downcasting", True)
    stat = stat.fillna(0).infer_objects(copy=False)

    dictate = dict()
    lobes = ["Occipital", "Parietal", "Frontal", "Temporal"]
    for lobe in lobes:
        temp = psd_intervals[psd_intervals.Lobe == lobe].drop(["Lobe"], axis=1)
        dictate[lobe] = return_signifcant(temp, df_intervals, print_debug=False)
        # print('dictate', dictate)
    return dictate


def convertsvg() -> None:
    import cairosvg
    import glob

    for file in glob.glob("*.svg"):
        name = file.split(".svg")[0]
        cairosvg.svg2png(url=name + ".svg", write_to=name + ".png")


def check_intervals(psd: pd.DataFrame, colbin: pd.Categorical, dataset: str, cols_to_drop: list[str] | None = None) -> None:
    # Drop non-numeric columns if specified
    if cols_to_drop is None:
        cols_to_drop = ["Region name", "Lobe"]

    cols_present = [col for col in cols_to_drop if col in psd.columns]
    if cols_present:
        psd = psd.drop(cols_present, axis=1)

    psd_mean = pd.DataFrame(psd.median(axis=0)).transpose()
    psd_intervals_mean = get_intervals(psd_mean, colbin)
    plot_intervals(psd_intervals_mean, 0, dataset)


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)




# Unsupervised classification of channels and no-peak set.
# %% [markdown]
# After reading the data output


# %% #for intracranial datata
#    import os
#    dir = os.path.dirname(__file__)
#    images = os.path.join(dir, '/images/')
#    if not os.path.exists(images):
#       os.makedirs(images)

