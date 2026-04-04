from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import collections as mc

from pesco.experimental.clusterization import (
    get_no_peak,
    identify_small_clusters,
    compute_clusters,
    cutintervals,
    get_intervals,
    return_signifcant,
    return_signifcant_lobes,
    ceildiv,
)

def plot_single_psd(psd_df: pd.DataFrame, f: np.ndarray, i: int) -> None:
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



def plot_psd_clusters(psd_df: pd.DataFrame, f: np.ndarray, dataset: str, smal: list[int] = [], nopeak: int | list[int] = [], output_dir: Path = Path("images")) -> tuple[Figure, Axes]:
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
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{dataset}_clusters.svg", format="svg")
    plt.show()
    return (fig, ax)


def plot_specific_clusterisation(
    psd: pd.DataFrame, f: np.ndarray, HowManyClusters: int, seed: int, dataset: str,
    ifall: bool = False, nopeak: int | list[int] = [], print_debug: bool = False,
    output_dir: Path = Path("images"),
) -> tuple[pd.DataFrame, list[int]]:
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
            fig, ax = plot_psd_clusters(psd_df, f, dataset, smal, nopeak, output_dir=output_dir)
    else:
        fig, ax = plot_psd_clusters(psd_df, f, dataset, smal, nopeak, output_dir=output_dir)
        print(nopeak) if print_debug else True
    print(psd_df["clusters"].value_counts()) if print_debug else True
    return psd_df, smal



def plot_subplot(temp: pd.DataFrame, median: np.ndarray, f: np.ndarray, dictate: dict[str, list] | Literal[False], lobe: str, ax: Axes | None = None, print_debug: bool = False) -> Axes:
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
    psd_clust: pd.DataFrame, psd: pd.DataFrame, f: np.ndarray, smal: list[int],
    dataset: str, dictate: dict[str, list] | Literal[False] = False,
    show: bool = False, print_debug: bool = False,
    output_dir: Path = Path("images"),
) -> None:
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

    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{dataset}_lobar_differences.svg", format="svg")
    if show:
        plt.show()
    else:
        plt.close()


def pipeline_lobe(dataset: str, f: np.ndarray, psd: pd.DataFrame, psd_clust: pd.DataFrame, smal: list[int], output_dir: Path = Path("images")) -> None:
    df, _ = get_no_peak(psd_clust, smal)
    colbin, _ = cutintervals(f)
    dictate = return_signifcant_lobes(df, psd, colbin)
    plot_lobes(psd_clust, psd, f, smal, dataset, dictate, show=True, output_dir=output_dir)


def pipeline_regions(dataset: str, f: np.ndarray, psd: pd.DataFrame, psd_clust: pd.DataFrame, smal: list[int], output_dir: Path = Path("images")) -> None:
    matplotlib.rcParams.update({"font.size": 8})
    df, median = get_no_peak(psd_clust, smal)
    colbin, _ = cutintervals(f)
    df_intervals = get_intervals(df, colbin)
    output_dir.mkdir(exist_ok=True)
    for iterate_lobe in psd["Lobe"].unique():
        regions = psd[psd.Lobe == iterate_lobe]["Region name"].unique()
        dictate = dict()
        how_long = ceildiv(len(psd[psd.Lobe == iterate_lobe]["Region name"].unique()), 2)
        fig, axes = plt.subplots(how_long, 2, figsize=(10, 8 * ceildiv(how_long, 2)))
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
                median, f, dictate, iterate_region, ax, print_debug=False,
            )
        fig.get_axes()[0].annotate(
            "Regional differences in " + iterate_lobe.lower() + " lobe (" + dataset + ")",
            (0.5, 0.95), xycoords="figure fraction", ha="center", fontsize=24,
        )
        plt.savefig(
            output_dir / f"{dataset}_{iterate_lobe}_regional_differences.svg",
            format="svg", bbox_inches="tight", pad_inches=0,
        )
        plt.show()
