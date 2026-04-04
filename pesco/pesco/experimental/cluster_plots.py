import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import matplotlib
from matplotlib import collections as mc

from pesco.experimental.clusterization import get_no_peak, identify_small_clusters, compute_clusters

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



def plot_psd_clusters(psd_df, f, dataset, smal=[], nopeak=[]):
    psd_medianas = psd_df.groupby("clusters").median()
    matplotlib.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for k in range(0, len(psd_medianas.index)):
        mediana = psd_medianas.loc[k]
        temp_label = (
            "cl. " + str(k) + " (" + str(psd_df["clusters"].value_counts().loc[k]) + " el.)"
        )
        if k in smal and k == nopeak:
            ax.semilogx(f, mediana, linewidth=4.0, color="black", label=temp_label)
        elif k == nopeak:
            ax.semilogx(f, mediana, linestyle=":", linewidth=5.0, label=temp_label)
        else:
            ax.semilogx(f, mediana, alpha=0.5, label=temp_label)
    ax.legend()
    ax.set_xticks([0.5, 4, 8, 13, 30, 80])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid()
    coordinates = [2, 6, 10, 16, 36]
    textes = [r"""$\delta$""", r"""$\theta$""", r"""$\alpha$""", r"""$\beta$""", r"""$\gamma$"""]
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
    smal = identify_small_clusters(psd_medianas)
    if print_debug:
        print()
        for index in smal:
            print()
            print("cluster", index, "when we have", HowManyClusters, "clusters")
    if ifall:
        for nopeak in range(0, HowManyClusters):
            fig, ax = plot_psd_clusters(psd_df, f, dataset, smal, nopeak)
    else:
        fig, ax = plot_psd_clusters(psd_df, f, dataset, smal, nopeak)
        print(nopeak) if print_debug else True
    print(psd_df["clusters"].value_counts()) if print_debug else True
    return psd_df, smal



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
