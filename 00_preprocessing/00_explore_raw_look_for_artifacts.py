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
 # To do:
 # * Use autoreject
 # remove artifacts and then plot data - how it change the results
# prepare email
# 
# %%
from IPython import get_ipython 
ip = get_ipython()
ip.run_line_magic('load_ext', 'autoreload')
ip.run_line_magic('autoreload', '2')
#ip.magic('%matplotlib inline ')
# %matplotlib inline

import pandas as pd
import mne as mne
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 5]
# %%
datafolder = '/home/daniel/PhD/notebooks/pesco-pipeline/data/preproc/'

df_ieeg = pd.read_csv(datafolder + "ieeg_raw.csv")
raw_ieeg = mne.io.read_raw_fif(datafolder +"ieeg_raw.fif",preload=True)


df_src = pd.read_csv(datafolder +"sources_raw.csv")
raw_src = mne.io.read_raw_fif(datafolder +"sources_raw.fif", preload=True)
# %%
scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
x= raw_ieeg.plot(n_channels=200, scalings=scalings, title='Auto-scaled Data from arrays',
         show=True, block=True)

# %% [markdown]
# The intracranial data timeseries contain "empty" spaces. where some arfifact was removed (problem for later).
#
# %%
import pathlib 
import numpy as np
from pesco.pesco import preprocess
PROJECT_PATH = '/home/daniel/PhD/' # set path to project
DATA_PATH = pathlib.Path(PROJECT_PATH)
raw, channels = preprocess.load_single_source(str(DATA_PATH)+ '/data/Mantini2018/', 1)

scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'

x= raw.plot(n_channels=72, scalings=scalings, title='Data from electrodes',
         show=True, block=True, lowpass=40)
#_ = raw.plot_psd(tmax=np.inf, average=False)
#raw , chan_details = preprocess.load_ieeg(str(DATA_PATH) + '/data/Frauscher2018/', datafolder)
#rtifacts = mne.preprocessing.find_eog_events(raw_src)
# %%
##
## make 1 second epochs
## Inspection by eye
##from mne.preprocessing.ica import corrmap  # noqa
#X = raw._data
#raw_tmp = raw.copy()
#raw_tmp.filter(1, None)
#ica = mne.preprocessing.ICA(method="extended-infomax", random_state=1)
#from mne.decoding import UnsupervisedSpatialFilter
#
#from sklearn.decomposition import PCA, FastICA
#pca = UnsupervisedSpatialFilter(PCA(30), average=False)
#pca_data = pca.fit_transform(X)
##ica.fit(raw_tmp)
## %%
#from mne.preprocessing import ICA
#method = 'fastica'
#
## Choose other parameters
#n_components = 25  # if float, select n_components by explained variance of PCA
#decim = 3  # we need sufficient statistics, not all time points -> saves time
#
## we will also set state of the random number generator - ICA is a
## non-deterministic algorithm, but we want to have the same decomposition
## and the same order of components each time this tutorial is run
#random_state = 23
#ica = ICA(n_components=n_components, method=method, random_state=random_state)
#print(ica)
#
#reject = dict(eeg=5e-12)
#ica.fit(raw, decim=decim, reject=reject)
#print(ica)
# %%
inds = raw_ieeg.time_as_index([40., 60.]) 
#data, times = raw[picks, t_idx[0]:t_idx[1]]
plt.plot(raw_ieeg.times[inds[0]:inds[1]], raw_ieeg._data[1, inds[0]:inds[1]])

# %% [markdown]
#
# Below I want to investigate  visible problems with "artifacts - spikes"in some channels in reconstructed sources data.  

# %%

# filter to see how it looks in a alpha bands
alpha_dat = raw_src.copy().filter(8, 12, fir_design='firwin')

# %%
times = [52., 57.]

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
inds = raw_src.time_as_index(times) 
#data, times = raw[picks, t_idx[0]:t_idx[1]]
ax1.plot(raw_src.times[inds[0]:inds[1]], raw_src._data[1, inds[0]:inds[1]])
#data, times = raw[picks, t_idx[0]:t_idx[1]]
ax2.plot(alpha_dat.times[inds[0]:inds[1]], alpha_dat._data[1, inds[0]:inds[1]])

#_ = alpha_dat.plot_psd(tmax=np.inf, average=False)


# sen this figure (also compute without artifact)

# %%
# Extract data from the first 5 channels, from 1 s to 3 s.
sfreq = raw_src.info['sfreq']
data, times = raw_src[3:7, int(sfreq * 60):int(sfreq * 120)]
_ = plt.plot(times, data.T)
_ = plt.title('Sample channels')

# %% [markdown]
#  # MNE channels plots
#  
#  ## Data description
#  Eyes closed in the intracranial dataset.

# %%
  
scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
x= raw_ieeg.plot(n_channels=100, scalings=scalings, title='Data from electrodes',
         show=True, block=True, lowpass=40)
_ = raw_ieeg.plot_psd(tmax=np.inf, average=False)


x= raw_src.plot(n_channels=100, scalings=scalings, title='Data from electrodes',
         show=True, block=True, lowpass=40)
_ = raw_src.plot_psd(tmax=np.inf, average=False)


# %%
## plot filtered fragmen of the signal from 70s to 210s

# %%
# subbset the data
data_crop = raw_src.copy().crop(70, 210)
#data, times = raw[picks, t_idx[0]:t_idx[1]]
x= data_crop.plot(n_channels=100, scalings=scalings, title='Data from electrodes',
         show=True, block=True, lowpass=40)
_ = data_crop.plot_psd(tmax=np.inf, average=False)
