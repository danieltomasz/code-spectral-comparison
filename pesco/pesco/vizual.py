# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Created on Mon Sep  3 13:08:50 2018

@author: daniel
"""

# %%
import numpy as np
import seaborn.timeseries


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

# flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
# sns.palplot(sns.color_palette(flatui))


def _plot_range_band(*args, central_data=None, ci=None, data=None, **kwargs):
    # upper = data.max(axis=0)
    # lower = data.min(axis=0)
    upper = np.quantile(data, 0.95, axis=0)
    lower = np.quantile(data, 0.05, axis=0)
    # import pdb; pdb.set_trace()
    ci = np.asarray((lower, upper))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})
    seaborn.timeseries._plot_ci_band(*args, **kwargs)


def _plot_range_bars(*args, central_data=None, ci=None, data=None, **kwargs):
    # upper = data.max(axis=0)
    # lower = data.min(axis=0)
    upper = np.quantile(data, 1, axis=0)
    lower = np.quantile(data, 0, axis=0)
    # import pdb; pdb.set_trace()
    ci = np.asarray((lower, upper))
    kwargs.update({"central_data": central_data, "ci": ci, "data": data})
    seaborn.timeseries._plot_ci_bars(*args, **kwargs)


def customPlot(*args, **kwargs):
    df = kwargs.pop("data")
    pivoted = df.pivot(index="subindex", columns="frequency", values="spectral_density")
    sns.tsplot(
        pivoted.values,
        time=np.arange(0.5, 80.5, 0.5),
        err_style="range_band",
        n_boot=0,
        color=kwargs["color"],
        estimator=np.median,
    )
    # sns.tsplot(pivoted.values, err_style="range_bars", n_boot=0, color=kwargs['color'])
    ax = sns.tsplot(
        pivoted.max(axis=0),
        time=np.arange(0.5, 80.5, 0.5),
        n_boot=0,
        color=kwargs["color"],
        linestyle="dashed",
        estimator=np.median,
    )
    ax = sns.tsplot(
        pivoted.min(axis=0),
        time=np.arange(0.5, 80.5, 0.5),
        n_boot=0,
        color=kwargs["color"],
        linestyle="dashed",
        estimator=np.median,
    )
    # ax.set_xscale('log')
    xposition = [4, 8, 13, 40]
    for xc in xposition:
        plt.axvline(x=xc, color="k", linestyle="--")
    # xticks = [ 4, 8, 13, 30,80]
    # ax.xticks(xticks, ticklabels)
    ax.set_xticks(xposition, minor=False)


sns.timeseries._plot_range_band = _plot_range_band

sns.timeseries._plot_range_bars = _plot_range_bars
