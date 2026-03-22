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

import pandas as pd
import mne as mne
import scipy
import numpy as np
from pesco.pesco import foofexplore
from pesco.pesco import spectral


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 5]

datafolder = '/home/daniel/PhD/notebooks/pesco-pipeline/data/preproc/'


# %%

df_ieeg = pd.read_csv(datafolder + "ieeg_raw.csv")
raw_ieeg = mne.io.read_raw_fif(datafolder +"ieeg_raw.fif")

df_sources = pd.read_csv(datafolder +"sources_raw.csv")
raw = mne.io.read_raw_fif(datafolder +"sources_raw.fif")
# %%
def get_psd(raw):
    Fs = raw.info["sfreq"]
    #copy = raw.copy().crop(tmin=0,tmax=300)
    copy= raw.copy()
    data, times = copy[:,:]
    #data, times = raw[:,:]
    #data_filt, notch_filtba = notch_default(data, 50, 2, Fs, 3)
    #data_filt, notch_filtba = notch_default(data_filt, 60, 2, Fs, 3)
    
    #    The spectral density in each channel was estimated with
    #    Welch’s method, i.e. averaging the magnitude of the discrete
    #    time Fourier transform of 59 overlapping blocks of 2 s dur-
    #    ation and 1 s step, weighted by a Hamming window. In each
    #    channel the resulting spectral density was normalized to a total
    #    power equal to one, making it independent of the EEG signal
    #    amplitude.
    #
    #    
    f, psd = scipy.signal.welch(data, Fs, nperseg=2*Fs,noverlap=Fs)
    psd_sum = np.sum(psd,1)
    psd_norm= psd/psd_sum[:,np.newaxis]
    psd_norm = psd_norm    
    psd_norm = pd.DataFrame(psd_norm,index=raw.info["ch_names"])
    psd_norm.columns = f
    plt.plot(f, np.mean(psd_norm))
    return (f, psd_norm)
# %%


f, psd1 = get_psd(raw_ieeg)

plt.plot(f, psd1[1,:])
# %%

#
#peaks_ieeg  = foofexplore.comp_foof(raw_ieeg , df_ieeg, datafolder + "ieeg_peaks_foofed.csv")
#peaks_sources  = foofexplore.comp_foof(raw_sources, df_sources, datafolder + "sources_peaks_foofed.csv")
#all_data= peaks_ieeg.append(peaks_sources)
#all_data.to_csv(datafolder+ 'all_data_peaks_foofed.csv')



# %%
# CHECJ THE 
# * FIND HIGHEest peak in the bands

# # Extract alpha oscillations - all channels
#alphas = get_band_peak_group(fg.get_all_data('peak_params'), [7, 14], len(fg))
# Plot the differently filtered traces - check for differences
# Filter data to FOOOF derived alpha band




# %%
# Packages I ams using to c
ip.magic('load_ext watermark')
ip.magic('watermark -a "Daniel Borek" -u -d -p mne,bycycle,fooof,pandas,numpy,seaborn,matplotlib,pesco,neurodsp') 

