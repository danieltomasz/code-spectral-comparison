# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocessing spectra
#
# This notebook shows how to prepare the data to further analysis. 
# *  load matlab file
# * prepare  spectra 
# * plot them
#
# Dataset properties:
# * All contacts were localized in a common stereotactic space allowing the accumulation and superposition of results from many subjects. 
# * Sixty-second artefact-free sections during wakefulness were selected. 
# * Power spectra were calculated for 38 brain regions, and compared to a set of channels with no spectral peaks in order to identify significant peaks in the different regions. 
# * A total of 1785 channels with normal brain activity from 106 patients were identified. There were on average 2.7 channels per cm3 of cortical grey matter. The number of contacts per brain region averaged 47 (range 6–178)
#
# * patients with refractory focal epilepsy are the only human subject  where extensive  intracranial cortical EEG studies are undertaken
#
#


# %%
from IPython import get_ipython 
ip = get_ipython()
ip.run_line_magic('load_ext', 'autoreload')
ip.run_line_magic('autoreload', '2')

ip.run_line_magic('load_ext', 'watermark')
ip.run_line_magic('watermark', '-a "Daniel Borek" -u -d -p mne,bycycle,fooof,pandas,numpy,pesco')

# %%
import numpy as np
import pandas as pd
import mne as mne

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import sys
sys.path.append("..")  #add folder levels above notebook

# this is custom kenel 
# print kernel name if running mne virtulenv

# %%
from pesco import preprocess as utils
raw , result = utils.load_ieeg()

# %%

raw = utils.filter_data(raw)
scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
x= raw.plot(n_channels=100, scalings=scalings, title='Auto-scaled Data from arrays',
         show=True, block=True)


# %%
sources= utils.load_sources()
sources = utils.filter_data(sources)
scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
y= sources.plot(n_channels=100, scalings=scalings, title='Auto-scaled Data from arrays',
         show=True, block=True)

# %% [markdown]
#  ### Plot the preprepocessed spectra of the signal 

# %%
#psd_df=pd.read_csv('../data/interim/power.csv',index_col=0)
#frequency_bins = np.loadtxt("../data/interim/frequency_bins.csv")
# scale for y
# add randomizing channels

plt.figure(figsize=(12, 10))
plt.subplot(211)
psd_df,frequency_bins = utils.get_psd(raw)

for i in range(0,100):
    plt.plot(frequency_bins, psd_df.values[i,:])
    plt.yscale('log')
plt.title('Spectras of electrodes')


src_df,frequency_bins = utils.get_psd(sources)
plt.subplot(212)

for i in range(0,100):
    plt.plot(frequency_bins, src_df.values[i,:])
    plt.yscale('log')
plt.title('Spectras of sources')


# %% [markdown]
# # Plot clusterization result

# %%
params = {"how_many_clusters":10 ,"max_iterations": 300, "n_initial": 100 , "random_seed": None}
src_label, medians = utils.clusterize_spectra(src_df, **params)
how_many_clusters=params["how_many_clusters"]

# %%
plt.figure(figsize=(22, 28))
plt.subplot(211)
for k in range(0, how_many_clusters):
    mean_cluster_spectrum = np.mean(src_df[src_label['clusters']==k]) 
    plt.plot(frequency_bins, mean_cluster_spectrum)
    plt.legend(range(0,how_many_clusters))
    plt.yscale('log')
    plt.xscale('log')

    plt.xlim(1, 40)
    xticks = [1, 2, 4, 8, 16, 32]
    ticklabels = ['1', '2', '4', '8', '16', '32']
    plt.xticks(xticks, ticklabels)

# %%
params = {"how_many_clusters":10 ,"max_iterations": 300, "n_initial": 100 , "random_seed": None}
psd_label, medians = utils.clusterize_spectra(psd_df, **params)
how_many_clusters=params["how_many_clusters"]

# %%
plt.figure(figsize=(22, 18))
plt.subplot(211)
for k in range(0, how_many_clusters):
    mean_cluster_spectrum = np.mean(psd_df[psd_label['clusters']==k]) 
    plt.plot(frequency_bins, mean_cluster_spectrum)
    plt.legend(range(0,how_many_clusters))
    plt.yscale('log')
    #plt.xscale('log')

    plt.xlim(1, 40)
    xticks = [1, 2, 4, 8, 16, 32]
    ticklabels = ['1', '2', '4', '8', '16', '32']
    plt.xticks(xticks, ticklabels)
#plot means of the clusters


plt.title('Median od clusters')

plt.subplot(212)

for k in range(0, how_many_clusters):
    mean_cluster_spectrum = np.mean(psd_df[psd_label['clusters']==k]) 
    gradient = np.gradient(mean_cluster_spectrum)
    plt.plot(frequency_bins, gradient)
    plt.legend(range(0,how_many_clusters))
    #plt.yscale('log')
    #plt.xscale('log')

    plt.xlim(1, 40)
    xticks = [1, 2, 4, 8, 16, 32]
    ticklabels = ['1', '2', '4', '8', '16', '32']
    plt.xticks(xticks, ticklabels)
plt.title('Gradients')

plt.show()

# %% [markdown]
# # Rand index 
# Rand index  R is a measure of the similarity between two data clusterings. It  is calculated by the formula
#
# $R = \frac{a + b}{{n}\choose{2}} $
#
# The $a$ in the formula refers to the number of   pairs of elements belonging to a same cluster across two different clustering results and the $b$ refers to the number of  pairs of elements  in different clusters across two different clustering results. The denominator is a binomial coefficient. ${n}\choose{2}$ is the number of unordered pairs in a set of n elements. For example, if you have set of 4 elements {a, b, c, d}, there are 6 unordered pairs: {a, b}, {a, c}, {a, d}, {b, c}, {b, d}, and {c, d}. 
#
#  Adjusted Rand index (ARI), is the corrected-for-chance (number of clusters) version of the Rand index. More to read [here](https://davetang.org/muse/2017/09/21/adjusted-rand-index/).

# %%
utils.print_smallest(medians)

# %%



