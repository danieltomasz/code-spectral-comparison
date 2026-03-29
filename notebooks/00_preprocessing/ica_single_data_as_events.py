# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#%%
#raw.info['dig']
from IPython import get_ipython 
ip = get_ipython()
ip.magic('load_ext autoreload')
ip.magic('autoreload 2')
ip.magic('%matplotlib inline')


# %%
import pathlib 
import numpy as np
from pesco.pesco import preprocess
import mne
mne.__version__

import autoreject
datafolder = '/home/daniel/PhD/notebooks/pesco-pipeline/data/ica/'


PROJECT_PATH = '/home/daniel/PhD/' # set path to project
DATA_PATH = pathlib.Path(PROJECT_PATH)
i = 1
#for i in range(1,2):
print('Data from subject ' + str(i))
raw, channels = preprocess.load_single_source(str(DATA_PATH)+ '/data/Mantini2018/', i)
scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'

raw.plot(n_channels=76, scalings=scalings, duration=20., title='Data from subject ' +    str(i), show=True, block=True, lowpass=40)
#%%
raw_filt = raw.filter(0.5, 15, n_jobs=4)


# %%
raw_tmp = raw.copy()
raw_tmp.filter(1, None)
ica = mne.preprocessing.ICA(method='picard', random_state=1)
ica.fit(raw_tmp)
#ica.plot_sources(inst=raw_tmp)
ica.detect_artifacts(raw_tmp) # bad component
ica.plot_sources(inst=raw_tmp,picks=ica.exclude,  start=0, stop=300.)
ica.plot_sources(inst=raw_tmp, picks=ica.exclude)
ica.save( datafolder + "subject" + str(i) + "-ica.fif")
#%%
raw_corrected = raw.copy()
ica.apply(raw_corrected)

y = raw_corrected.plot(n_channels=76, scalings=scalings, title='Data from subject ' + str(i),
         show=True, block=True, lowpass=40)
#%%



