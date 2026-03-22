# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from IPython import get_ipython 
ip = get_ipython()
ip.magic('load_ext autoreload')
ip.magic('autoreload 2')

# %matplotlib inline

import pathlib
import pandas as pd
from  pesco.pesco import preprocess


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 5]

# %%

PROJECT_PATH = '/home/daniel/PhD/' # set path to project
path = pathlib.Path(PROJECT_PATH)

raw , chan_details = preprocess.load_sources(str(path) + '/data/Mantini2018/')
#raw2.save('sample_audvis_meg_trunc_raw.fif', tmin=0, tmax=150, picks=picks,         overwrite=True)


# %%
# Compute segments
tmin, tmax = 10, 130

#def comp_foof_seg(raw, chan_details, tmin, tmax):

import pandas as pd
import numpy as np
from neurodsp.spectral import compute_spectrum
from fooof import FOOOFGroup

import  mne 

fmin, fmax = 3, 80
# PSD settings
srate = raw.info['sfreq']
n_fft, n_overlap, n_per_seg = int(2*srate), int(srate), int(2*srate)
data, times = raw[:]  
Fs= raw.info['sfreq']
#freqs, spectrum = compute_spectrum(data, Fs,  method='median', nperseg=2*Fs)
spectrum,freqs= mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, tmin=tmin ,tmax=tmax,
                                       n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, verbose=False)
psd_df = pd.DataFrame(spectrum, index=chan_details["region_number"])
fg = FOOOFGroup(peak_width_limits=[1, 8],min_peak_amplitude=0.05, max_n_peaks=6)
fg.fit(freqs, psd_df.values, freq_range=[3, 40],n_jobs=-1)

#extracting data from  fg 
temp_df = pd.DataFrame() # prepare error and slope dataframe
temp_df['sls'] = fg.get_all_data('background_params', 'slope')
temp_df['errors']= fg.get_all_data('error')
temp_df['r2s']=fg.get_all_data('r_squared')
temp_df['ch_number'] = np.arange(len(chan_details))

peaks = fg.get_all_data('peak_params') # prepare peaks dataframe
peaks_df = pd.DataFrame(peaks)
peaks_df.columns = ['CF', 'Amp', 'BW', 'ch_number']

chan_details['ch_number'] = np.arange(len(chan_details)) # prepare ch_details 
peaks_all=chan_details.set_index('ch_number').join(peaks_df.set_index('ch_number')).join(temp_df.set_index('ch_number'))
#return peaks_all



# Data settings

#peaks_sources  = comp_foof_seg(raw2 , chan_details2, tmin, tmax)

# %%
X = np.arange(2, int(data.shape[1] /srate - 60), 30)
Y = X + 60