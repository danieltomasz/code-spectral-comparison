import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import matplotlib
from matplotlib import collections as mc

from pesco.experimental.clusterization import get_no_peak

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
