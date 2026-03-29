"""
.. _tut_creating_data_structures:

Creating MNE's data structures from scratch
===========================================

MNE provides mechanisms for creating various core objects directly from
NumPy arrays.
"""

import mne
import numpy as np



n_channels = 32
sampling_rate = 200
info = mne.create_info(n_channels, sampling_rate)
print(info)

###############################################################################
# You can also supply more extensive metadata:

# Names for each channel
channel_names = ['MEG1', 'MEG2', 'Cz', 'Pz', 'EOG']

# The type (mag, grad, eeg, eog, misc, ...) of each channel
channel_types = ['grad', 'grad', 'eeg', 'eeg', 'eog']

# The sampling rate of the recording
sfreq = 1000  # in Hertz

# The EEG channels use the standard naming strategy.
# By supplying the 'montage' parameter, approximate locations
# will be added for them
montage = 'standard_1005'

# Initialize required fields
info = mne.create_info(channel_names, sfreq, channel_types, montage)

# Add some more information
info['description'] = 'My custom dataset'
info['bads'] = ['Pz']  # Names of bad channels

print(info)

data = np.random.randn(5, 1000)

# Initialize an info structure
info = mne.create_info(
    ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
    ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
    sfreq=100
)

custom_raw = mne.io.RawArray(data, info)
print(custom_raw)

sfreq = 100
data = np.random.randn(10, 5, sfreq * 2)

# Initialize an info structure
info = mne.create_info(
    ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
    ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
    sfreq=sfreq
)

events = np.array([
    [0, 0, 1],
    [1, 0, 2],
    [2, 0, 1],
    [3, 0, 2],
    [4, 0, 1],
    [5, 0, 2],
    [6, 0, 1],
    [7, 0, 2],
    [8, 0, 1],
    [9, 0, 2],
])


event_id = {'Auditory/Left': 1, 'Auditory/Right': 2}


tmin = -0.1


epochs  = mne.EpochsArray(data, info, events, tmin, event_id)


# %%
import mne
import autoreject
from functools import partial
from autoreject import AutoReject


picks = mne.pick_types(epochs.info, meg=False, eeg=True, stim=False, eog=False)  # Find indices of all EEG channels
thresh_func = partial(autoreject.compute_thresholds, picks=picks, method='random_search')

n_interpolates = np.array([1, 4, 16])
consensus_percs = np.linspace(0, 1.0, 11)
#data.info['projs'] = list()  # remove proj, don't proj while interpolating

ar = AutoReject(n_interpolates, consensus_percs, picks=picks,
                thresh_method='random_search', random_state=42)

# Note that fitting and transforming can be done on different compatible
# portions of data if needed.
ar.fit(epochs['Auditory/Left'])
epochs_clean = ar.transform(epochs['Auditory/Left'])
evoked_clean = epochs_clean.average()
evoked = epochs['Auditory/Left'].average()#ar.fit(epochs)
#epochs = ar.transform(epochs)