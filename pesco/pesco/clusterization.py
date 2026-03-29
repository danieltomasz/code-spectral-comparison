#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% [markdown]
# # Frequency peaks in intracranial and  reconstructed sources data
#
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import collections as mc
import sklearn
import mpltex

# import pesco.utils as utils
import pesco.preprocess as preprocess

from scipy import stats

seed = 3

# for importing r

# %%


def get_no_peak(psd_clust, smal):
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


def plot_subplot(temp, median, f, dictate, lobe, ax=None, print_debug=False):
    if ax is None:
        ax = plt.gca()
    q75 = temp.quantile(0.75, axis=0)
    q25 = temp.quantile(0.25, axis=0)
    ax.semilogx(f, median, color="black")
    ax.semilogx(f, q75, color="pink")
    ax.semilogx(f, q25, color="pink")
    ax.semilogx(f, temp.median(0), color="red")

    ax.semilogx(f, temp.max(axis=0), color="r", linestyle=":")
    ax.semilogx(f, temp.min(axis=0), color="r", linestyle=":")
    ax.fill_between(f, q25, q75, facecolor="pink", interpolate=True)
    # ax.legend();

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid()

    # textes greek letters

    if dictate:
        lines = dictate[lobe]
        print(lines) if print_debug else True
        lc = mc.LineCollection(lines, linewidths=2)
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Normalized spectral density")
    ax.set_title(lobe + " lobe - " + str(len(temp)) + " channels")
    coordinates = [2, 6, 10, 16, 36]
    textes = [
        r"""$\delta$""",
        r"""$\theta$""",
        r"""$\alpha$""",
        r"""$\beta$""",
        r"""$\gamma$""",
    ]
    for t, text in zip(coordinates, textes):
        ax.text(t, 0.085, text, fontsize=14)
    ax.set_xticks([0.5, 4, 8, 13, 30, 80])
    ax.set_ylim(0, 0.10)
    ax.set_xlim(0.5, 80)
    return ax


def plot_lobes(
    psd_clust, psd, f, smal, dataset, dictate=False, show=False, print_debug=False
):

    # plt.semilogx(f,median)
    # plt.show()
    _, median = get_no_peak(psd_clust, smal)
    matplotlib.rcParams.update({"font.size": 22})
    lobes = ["Occipital", "Parietal", "Frontal", "Temporal"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle("Lobar differences in EEG frequences: " + dataset)
    fig.subplots_adjust(hspace=0.25)
    fig.subplots_adjust(wspace=0.25)

    for ax, lobe in zip(axes.flatten(), lobes):
        temp = psd[psd.Lobe == lobe].drop(["Region name", "Lobe"], axis=1)
        plot_subplot(temp, median, f, dictate, lobe, ax=ax, print_debug=False)

    plt.savefig("images/" + dataset + "_lobar_differences.svg", format="svg")
    if show:
        plt.show()
    else:
        plt.close()


# plot by lobe
def cutintervals(x):
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


def get_intervals(psd, colbin):
    dictionary = dict(zip(psd.columns, colbin))
    psd_intervals = psd.T.groupby(dictionary).sum().T
    psd_intervals.columns = psd_intervals.columns.astype(str, copy=False)
    return psd_intervals


def plot_intervals(psd_intervals, i, dataset):

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


def plot_ecdf(v1, v2, lobe, idxCol, l):
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
        "images/" + lobe + "_" + str(l[0]) + "_" + str(l[1]) + "_interval.svg",
        format="svg",
    )


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def return_signifcant(temp, df_intervals, print_debug=False):
    import re
    import rpy2
    from rpy2.robjects.packages import importr

    rstats = importr("stats")
    list_of_intervals = []
    for (idxCol, v1), (_, v2) in zip(temp.items(), df_intervals.items()):
        print(v1, v2, idxCol) if print_debug else True
        # v1 is a column from lobes
        # v2 is  a column from no peak set
        t, pval = stats.ks_2samp(v1, v2)
        # details of r function https://rdrr.io/cran/dgof/man/ks.test.html
        # ther is no computation of True value only assymptotic aproximation
        v1 = rpy2.robjects.vectors.FloatVector(v1)
        v2 = rpy2.robjects.vectors.FloatVector(v2)
        htest = rstats.ks_test(v1, v2, alternative="less", exact=False)
        htestlist = list(htest)
        t = htestlist[0][0]
        pval = htestlist[1][0]
        pval = pval * 22 * 42
        p = 0.05
        y = 0.08
        if pval < p:
            p = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")
            l = p.findall(idxCol)
            l = [float(i) for i in l]
            list_of_intervals.append([(l[0], y), (l[1], y)])
            # print('list of intervals', *list_of_intervals, sep = ", ")
            print(idxCol, t, format(pval, ".5f")) if print_debug else True
            # plot_ecdf(v1, v2, lobe, idxCol, l)
            # i = i + 1
    return list_of_intervals


def return_signifcant_lobes(df, psd, colbin, print_debug=False):

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


def convertsvg():
    import cairosvg
    import glob

    for file in glob.glob("*.svg"):
        name = file.split(".svg")[0]
        cairosvg.svg2png(url=name + ".svg", write_to=name + ".png")


def check_intervals(psd, colbin, dataset, cols_to_drop=None):
    # Drop non-numeric columns if specified
    if cols_to_drop is None:
        cols_to_drop = ["Region name", "Lobe"]

    cols_present = [col for col in cols_to_drop if col in psd.columns]
    if cols_present:
        psd = psd.drop(cols_present, axis=1)

    psd_mean = pd.DataFrame(psd.median(axis=0)).transpose()
    psd_intervals_mean = get_intervals(psd_mean, colbin)
    plot_intervals(psd_intervals_mean, 0, dataset)


def ceildiv(a, b):
    return -(-a // b)


def pipeline_lobe(dataset, f, psd, psd_clust, smal):
    df, median = get_no_peak(psd_clust, smal)
    colbin, _ = cutintervals(f)

    dictate = return_signifcant_lobes(df, psd, colbin)
    plot_lobes(psd_clust, psd, f, smal, dataset, dictate, show=True)


mpltex.presentation_decorator


def pipeline_regions(dataset, f, psd, psd_clust, smal):
    matplotlib.rcParams.update({"font.size": 8})
    df, median = get_no_peak(psd_clust, smal)
    colbin, y = cutintervals(f)
    df_intervals = get_intervals(df, colbin)
    for iterate_lobe in psd["Lobe"].unique():
        regions = psd[psd.Lobe == iterate_lobe]["Region name"].unique()
        dictate = dict()
        how_long = ceildiv(
            len(psd[psd.Lobe == iterate_lobe]["Region name"].unique()), 2
        )
        fig, axes = plt.subplots(how_long, 2, figsize=(10, 8 * ceildiv(how_long, 2)))
        # fig.suptitle('Regional differences in '+ iterate_lobe  + 'in' +  dataset )
        fig.subplots_adjust(hspace=0.25)
        fig.subplots_adjust(wspace=0.25)
        for ax, iterate_region in zip(axes.flatten(), regions):
            temp_region = psd[
                (psd.Lobe == iterate_lobe) & (psd["Region name"] == iterate_region)
            ]
            temp_intervals = get_intervals(
                temp_region.drop(["Region name", "Lobe"], axis=1), colbin
            )
            dictate[iterate_region] = return_signifcant(
                temp_intervals, df_intervals, print_debug=False
            )
            plot_subplot(
                temp_region.drop(["Region name", "Lobe"], axis=1),
                median,
                f,
                dictate,
                iterate_region,
                ax,
                print_debug=False,
            )

        # fig.tight_layout()
        fig.get_axes()[0].annotate(
            "Regional differences in "
            + iterate_lobe.lower()
            + " lobe ("
            + dataset
            + ")",
            (0.5, 0.95),
            xycoords="figure fraction",
            ha="center",
            fontsize=24,
        )
        plt.savefig(
            "images/"
            + dataset
            + "_"
            + iterate_lobe
            + "_"
            + "_regional_differences.svg",
            format="svg",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.show()


# Unsupervised classification of channels and no-peak set.
# %% [markdown]
# After reading the data output


# %% #for intracranial datata
#    import os
#    dir = os.path.dirname(__file__)
#    images = os.path.join(dir, '/images/')
#    if not os.path.exists(images):
#       os.makedirs(images)

# %%
# importing dataset and finding cluserisation
if __name__ == "__main__":
    dataset = "intracranial data"
    f, psd = preprocess.prepare_psd()
    psd_clust, smal = preprocess.plot_specific_clusterisation(
        psd.drop(["Region name", "Lobe"], axis=1),
        f,
        8,
        seed,
        dataset=dataset,
        ifall=False,
        nopeak=1,
    )
    # %%
    colbin, _ = cutintervals(f)
    check_intervals(psd, colbin, dataset)
    pipeline_lobe(dataset, f, psd, psd_clust, smal)
    pipeline_regions(dataset, f, psd, psd_clust, smal)

    # %%
    dataset = "reconstructed sources"
    DATA_PATH = "/home/daniel/PhD/data/Mantini2018"
    datafolder = "/home/daniel/PhD/notebooks/pesco-pipeline/data/preproc/"
    raw_src, result = preprocess.load_sources(DATA_PATH)
    f, psd_source_clust = preprocess.get_psd_mat(
        raw_src._data,
        raw_src.info["sfreq"],
        raw_src.info["ch_names"],
        name="power_ieeg.csv",
        save_psd=False,
    )
    Y = psd_source_clust.join(result)
    psd_clust, smal = preprocess.plot_specific_clusterisation(
        Y.drop(["Region name", "Lobe", "region_number", "dataset"], axis=1),
        f,
        6,
        seed,
        dataset=dataset,
        ifall=False,
        nopeak=4,
    )
    psd = Y.drop(["region_number", "dataset"], axis=1)

    # %%
    check_intervals(psd, colbin, dataset)
    pipeline_lobe(dataset, f, psd, psd_clust, smal)
    pipeline_regions(dataset, f, psd, psd_clust, smal)
#    pipeline(dataset, f, psd, psd_clust, smal)
# %% % plot per regions significant areas per regions

# Subplots are organized in a Rows x Cols Grid
# Tot and Cols are known
