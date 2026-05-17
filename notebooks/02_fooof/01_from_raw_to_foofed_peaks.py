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

# %%
# This file is for producing  foofed peaks

# %%
from IPython import get_ipython 
ip = get_ipython()
ip.magic('load_ext autoreload')
ip.magic('autoreload 2')

# %matplotlib inline

import pathlib
import pandas as pd
from  pesco.pesco import preprocess
import mne as mne

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 5]
# %% [markdown]
# # Read  and plot data
#
# Below I read the raw data from files. There are two data sets:  electrophysiological atlas <cite data-cite="6114165/24GR3L4D"></cite> and reconstructed sources  from  19 subjects (done by Jessica Samogin) Reconstructed sources contain 38 regions, based on regions from <cite data-cite="6114165/24GR3L4D"></cite> paper. I read all the mat files as  put them together as one single file per dataset.
#
#

# %%
datafolder = '/home/daniel/PhD/notebooks/pesco-pipeline/data/preproc/'
CREATEDATA  = 0 
if CREATEDATA == 1:
    PROJECT_PATH = '/home/daniel/PhD/' # set path to project
    path = pathlib.Path(PROJECT_PATH)
    
    raw , chan_details = preprocess.load_ieeg(str(path) + '/data/Frauscher2018/', datafolder)
    raw2 , chan_details2 = preprocess.load_sources(str(path) + '/data/Mantini2018/',datafolder)


# %%

df_ieeg = pd.read_csv(datafolder + "ieeg_raw.csv")
raw_ieeg = mne.io.read_raw_fif(datafolder +"ieeg_raw.fif")


df_sources = pd.read_csv(datafolder +"sources_raw.csv")
raw_sources = mne.io.read_raw_fif(datafolder +"sources_raw.fif")


# %%

from pesco.pesco.experimental import foofexplore

peaks_ieeg  = foofexplore.comp_foof(raw_ieeg , df_ieeg, datafolder + "ieeg_peaks_foofed.csv")
peaks_sources  = foofexplore.comp_foof(raw_sources, df_sources, datafolder + "sources_peaks_foofed.csv")
all_data= peaks_ieeg.append(peaks_sources)
all_data.to_csv(datafolder+ 'all_data_peaks_foofed.csv')



# %%
# CHECJ THE 
# * FIND HIGHEest peak in the bands

# 



# %%
# Packages I ams using to c
ip.magic('load_ext watermark')
ip.magic('watermark -a "Daniel Borek" -u -d -p mne,bycycle,fooof,pandas,numpy,seaborn,matplotlib,pesco,neurodsp') 

