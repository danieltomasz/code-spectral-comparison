# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.0.0-rc3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Load example data to use for this tutorial (a single example power spectrum)

# %%
from IPython import get_ipython 
ip = get_ipython()
ip.magic('load_ext autoreload')
ip.magic('autoreload 2')
#ip.magic('%matplotlib inline ')
# %matplotlib inline

import pandas as pd
import mne as mne
import numpy as np
from numpy.lib.stride_tricks import as_strided

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 5]

# %%
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    # samples = np.append(np.zeros(np.floor(frameSize / 2.0)), sig)
    samples = np.array(sig, dtype='float64')
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    # samples = np.append(samples, np.zeros(frameSize))
    frames = as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


def print_mat_nested(d, indent=0, nkeys=0):
    """Pretty print nested structures from .mat files   
    Inspired by: `StackOverflow <http://stackoverflow.com/questions/3229419/pretty-printing-nested-dictionaries-in-python>`_
    """
    
    # Subset dictionary to limit keys to print.  Only works on first level
    if nkeys>0:
        d = {k: d[k] for k in  list(d.keys())[:nkeys]}  # Dictionary comprehension: limit to first nkeys keys.

    if isinstance(d, dict):
        for key, value in d.items():         # iteritems loops through key, value pairs
          print( '\t' * indent + 'Key: ' + str(key))
          print_mat_nested(value, indent+1)

    if isinstance(d,np.ndarray) and d.dtype.names is not None:  # Note: and short-circuits by default
        for n in d.dtype.names:    # This means it's a struct, it's bit of a kludge test.
            print('\t' * indent + 'Field: ' + str(n))
            print_mat_nested(d[n], indent+1)
# %%
datafolder = '/home/daniel/PhD/notebooks/pesco-pipeline/data/preproc/'

df_ieeg = pd.read_csv(datafolder + "ieeg_raw.csv")
raw_ieeg = mne.io.read_raw_fif(datafolder +"ieeg_raw.fif",preload=True)
import scipy.io
mat = scipy.io.loadmat('/home/daniel/PhD/data/Frauscher2018/WakefulnessMatlabFile.mat')
mat.items()

print_mat_nested(mat, indent=1, nkeys=30)
patient= mat['Patient']
ChannelName= [l.flatten()[0] for l in mat['ChannelName'].flatten()]
filtered_list = [ChannelName[i] for i in range(len(ChannelName)) if (patient==1)[i]]
specific_chans = raw_ieeg.copy().pick_channels(filtered_list)
#from itertools import compress
#temp = list(compress(ChannelName, (patient==1)))
#temp=ChannelName[patient==1,]

df = pd.DataFrame(patient, columns=['Patient'])
df['ChannelName'] = mat['ChannelName']
df['Data'] =[ mat['Data'].T]

#df_src = pd.read_csv(datafolder +"sources_raw.csv")
#raw_src = mne.io.read_raw_fif(datafolder +"sources_raw.fif", preload=True)
#event_id =1 
#events = np.array([[200, 0, event_id],
#                   [1200, 0, event_id],
#                   [2000, 0, event_id]])  # List of three arbitrary events
## %%
#
#points= np.arange(0,raw_ieeg._data.shape[1],400)
#events= np.vstack([points,np.zeros(len(points)), np.zeros(len(points))+event_id])
#events = events.T
# %%
duration = 2.

# create a fixed size events array
# start=0 and stop=None by default
events = mne.make_fixed_length_events(raw_ieeg, event_id, duration=duration)
print(events)

# for fixed size events no start time before and after event
tmin = 0.
tmax = .99  # inclusive tmax, 1 second epochs

# create :class:`Epochs <mne.Epochs>` object
epochs = mne.Epochs(raw_ieeg, events=events, event_id=event_id, tmin=tmin,tmax=tmax,                     baseline=None, verbose=True, preload=True, flat=dict(ecog=0))
z= epochs.plot(scalings='auto', block=False)
epochs.plot_psd(fmin=2., fmax=40.,)

# %%
from autoreject import AutoReject
ar = AutoReject()
epochs_clean = ar.fit_transform(epochs)  

# %%
duration = 2.

# create a fixed size events array
# start=0 and stop=None by default
events = mne.make_fixed_length_events(raw_ieeg, event_id, duration=duration)
print(events)

# for fixed size events no start time before and after event
tmin = 0.
tmax = 1.99  # inclusive tmax, 1 second epochs

# create :class:`Epochs <mne.Epochs>` object
epochs = mne.Epochs(specific_chans, events=events, event_id=event_id, tmin=tmin,tmax=tmax,                     baseline=None, verbose=True, preload=True, flat=dict(ecog=0))
z= epochs.plot(scalings='auto', block=False)
epochs.plot_psd(fmin=2., fmax=40.)

# %%
#numpy.std(rolling_window(observations, n), 1)
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

a = np.array([1, 2, 3, 4, 5, 6]); rolling_window(a, 3)
np.lib.stride_tricks.as_strided(a, shape=(4,6), strides=(8,4))


# %%
Fs = raw_ieeg.info["sfreq"]
window_size = 2*Fs  # 2048-sample fourier windows
stride = 1/2*Fs       # 512 samples between windows
wps = fs/float(512) # ~86 windows/second
Xs = np.empty([int(10*wps),2048])

for i in range(Xs.shape[0]):
    Xs[i] = np.abs(fft(X[i*stride:i*stride+window_size]))
# %%
