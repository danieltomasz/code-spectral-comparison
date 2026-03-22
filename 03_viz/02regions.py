#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for reproducing  Lobar differences in EEG frequences (Frauscher et al., 2018)
Created on Tue Aug 28 12:41:13 2018

@author: Daniel Borek
"""

#import numpy as np
import pandas as pd
import mne as mne

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import sys
sys.path.append("..")  #add folder levels above notebook

from pesco import utils
from pesco import viz


#%%

raw1, lobes1 = utils.load_ieeg()
raw2, lobes2 = utils.load_sources()
#raw1 = utils.filter_data(raw1)

result_long1=utils.lobes_long(raw1, lobes1)
result_long2=utils.lobes_long(raw2, lobes2)
result=pd.concat([result_long1,result_long2], axis=0)

#raw = utils.filter_data(raw)

#%%
scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
x= raw1.plot(n_channels=100, scalings=scalings, title='Auto-scaled Data from arrays',
         show=True, block=True)


#%%
datatoplot = result

#cluster_overload = pd.read_csv("TSplot.csv", delim_whitespace=True)
datatoplot['subindex'] = datatoplot.groupby(['Lobe','frequency']).cumcount()


g = sns.FacetGrid(datatoplot, row="Lobe", col= 'dataset', sharey=False,  aspect=3)
g = (g.map_dataframe(viz.customPlot, 'frequency', 'spectral_density','subindex')
       .add_legend()
       .set(xticks=[ 4, 8, 13, 30,80],
            xscale="log"))

#g.add_legend();

