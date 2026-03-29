#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Mon Jan 14 16:18:21 2019

@author: daniel
"""

# %% [markdown]
# # Distribution of the peaks (fooof)

# %%



# %%

#all_data.rename(columns={'Region name':'reg'}, inplace=True)

# %%

import pandas as pd
import mne as mne
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
plt.close("all")

from pesco.pesco import foofexplore
datafolder = '/home/daniel/PhD/notebooks/pesco-pipeline/data/preproc/'

all_data = pd.read_csv(datafolder+ 'all_data_peaks_foofed.csv')

# %% [markdown]
# ## By region

# %%

# PRINT BY REGION
g = sns.FacetGrid(all_data, col="dataset",  row="Region name")
g = g.map(plt.scatter, "CF", "Amp", edgecolor="w")
# %%
fig = g.fig
fig.savefig('lobes.png') 

# %% [markdown]
# ## By lobe

# %%

# PRINT BY lobe


g = sns.FacetGrid(all_data, col="dataset", hue="Region name",  row="Lobe")
g = (g.map(plt.scatter, "CF", "Amp",s=50, alpha=.5, linewidth=.5, edgecolor="white", marker=".").add_legend())
# %% [markdown]
# ## Slope

# %%

# PRINT by slope


g = sns.FacetGrid(all_data, col="dataset",   row="Lobe")
g = (g.map(sns.boxplot, "Region name","sls").add_legend())


# %% [markdown]
# ## Compute  peaks for time segment


# %%
df_sources = pd.read_csv(datafolder +"sources_raw.csv")
raw_src = mne.io.read_raw_fif(datafolder +"sources_raw.fif")

# Data settings
data_crop = raw_src.copy().crop(10, 290)
tmin, tmax = 10, 130
tmins= list(range(0, 240, 30))
for u in tmins:
    print(u)
    
    peaks_sources  = foofexplore.comp_foof_seg(data_crop , df_sources, u, u + 30)



    #g = sns.FacetGrid(peaks_sources, col="dataset",  row="Region name")
    #g = g.map(plt.scatter, "CF", "Amp", edgecolor="w")



    g = sns.FacetGrid(peaks_sources, col="dataset", hue="Region name",  row="Lobe")
    g = (g.map(plt.scatter, "CF", "Amp",s=50, alpha=.5, linewidth=.5, edgecolor="white", marker=".").add_legend())
    # Calculate PSDs (across all channels) - from the first 2 minute of data
    #https://martinos.org/mne/stable/generated/mne.time_frequency.psd_welch.html
    plt.title( "30 s fragment starting from " + str(u) +  " s")




# %%


# %% [markdown] {"toc-hr-collapsed": true}
# # Canonical alpha and individual alpha
#
# compare canonical alpha (8-12) and foofed alpha (highest amplitude peak form 7-14 range (wide alpha +/-2 hz band)

# %% [markdown] {"toc-hr-collapsed": false}
# # Summary
# ## Report: questions
#
# * Some of the reconstructed channels are noisy, Espacially channel 1 (write specific subject name and channel). What could be the cause of that?
# * Also  
# * From the praveiling of alpha, the data is representing eye close 
#
# ## To do
#
# * Per channel distitbution of the  biggest peaks for source reconstructed dataset
# *  check different method for PSD computation

# %% [markdown]
# ## print packages versions I used 
