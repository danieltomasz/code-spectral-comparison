# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
Created on Wed Dec 12 17:06:17 2018

@author: daniel
"""
import pathlib
import pandas as pd
from autoreject import AutoReject
import mne
import numpy as np


PROJECT_PATH = '/home/daniel/PhD/' # set path to project
path = pathlib.Path(PROJECT_PATH)


# %%
from  pesco.pesco import preprocess
raw, result  = preprocess.load_single_source(PROJECT_PATH +"data/Mantini2018/", 1)

data= raw.copy().crop(1., 20.)
event_time=np.arange(200, data.n_times, 400)
code=np.zeros(len(event_time))
code[::2] = 2
code[1::2] = 1
events= np.array([event_time,np.zeros(len(event_time)), code]).T
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}
tmin, tmax = -0.2, 0.5


scalings = 'auto'
events= np.array([event_time,np.zeros(len(event_time)), code]).T
data.plot(events=events, n_channels=25, scalings=scalings, title='Auto-scaled Data from arrays',       show=True, block=True)
# %%
k = data.plot_psd(picks= np.arange(0,15,1),fmin=3, fmax=40)

# %%
tmin, tmax = -1, 1
events_id = dict(odd=1, even=2)
epochs = mne.Epochs(data, events.astype(int), event_id, tmin, tmax, proj=True,
                       baseline=None,                       preload=True, reject=None )


data.info['bads'] = []
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       include=[], exclude=[])
#%%


# data.info['bads'] = []
# picks = mne.pick_types(data.info, meg=False, eeg=True, stim=False, eog=False,
#                       include=[], exclude=[])
# ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
#                thresh_method='random_search', random_state=42)
#
# ar.fit(epochs['odd'])
# epochs_clean = ar.transform(epochs['odd'])
# evoked_clean = epochs_clean.average()
# evoked = epochs['odd'].average()
#
#
# ar.get_reject_log(epochs['Auditory/Left']).plot()


n_interpolates = np.array([1, 4, 8])
consensus_percs = np.linspace(0, 1.0, 11)
raw.info['projs'] = list()  # remove proj, don't proj while interpolating

ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
                thresh_method='bayesian_optimization', random_state=42, verbose = 'tqdm')

# Note that fitting and transforming can be done on different compatible
# portions of data if needed.
ar.fit(epochs['Auditory/Left'])
epochs_clean = ar.transform(epochs['Auditory/Left'])
evoked_clean = epochs_clean.average()
evoked = epochs['Auditory/Left'].average()

    # Same `dev_head_t` for all runs so that we can concatenate them.


# raw2 , chan_details2 = preprocess.load_sources(str(path) + '/data/Mantini2018/')
#
# event_id = dict(odd=1, even=2)
# tmin = -1
# info =data.info
#
# custom_epochs = mne.EpochsArray(data, info, events, tmin, event_id)
# mne.viz.plot_events(events, data.info['sfreq'], data.first_samp,                   event_id=event_id)
