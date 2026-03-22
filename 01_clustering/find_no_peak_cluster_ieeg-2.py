#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% [markdown]
# # Look for clusters
#
# %%
import pandas as pd
import pesco.pesco as pesco 
import pesco.pesco.preprocess as preprocess
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import collections  as mc
import sklearn
#import pesco.utils as utils
from scipy import stats
seed = 3

#cluster 3, seed 3


#%%
def get_no_peak(psd_clust, smal):
    no_peak = psd_clust[psd_clust["clusters"]==smal[0]]
    no_peak = no_peak[no_peak.columns[:-1]]
    median = np.median(no_peak, 0)
    temp_med = np.tile(median, (len(no_peak),1))
    no_peak["distances_to_median"] = sklearn.metrics.pairwise.paired_distances(no_peak, temp_med)
    df = no_peak[ no_peak.distances_to_median < no_peak.distances_to_median.quantile(.50)]
    return df , median

def plot_lobes(psd_clust,psd, f,  smal,  dataset,dictate=False, show = False):

    #plt.semilogx(f,median)
    #plt.show()
    df , median= get_no_peak(psd_clust, smal)
    matplotlib.rcParams.update({'font.size': 22})
    lobes = ['Occipital', 'Parietal',  'Frontal', 'Temporal']
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('Lobar differences in EEG frequences: ' + dataset )
    fig.subplots_adjust(hspace=0.25)
    fig.subplots_adjust(wspace=0.25)

    for ax, lobe in zip(axes.flatten(), lobes):
        temp = psd[psd.Lobe==lobe].drop(['Region name', 'Lobe'], axis=1)
        q75= temp.quantile(.75, axis=0)
        q25= temp.quantile(.25, axis=0)
        ax.semilogx(f,median, color='black')
        ax.semilogx(f,q75, color='pink')
        ax.semilogx(f,q25, color='pink')
        ax.semilogx(f,temp.median(0), color='red')
        
        ax.semilogx(f,temp.max(axis=0),color='r',linestyle=':')
        ax.semilogx(f,temp.min(axis=0),color='r',linestyle=':')
        ax.fill_between(f, q25, q75, facecolor='pink', interpolate=True)
        #ax.legend();
    
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid()
        
        # textes greek letters
        coordinates = [2,6,10, 16,36]
        textes = [ r'''$\delta$''', r'''$\theta$''', r'''$\alpha$''', r'''$\beta$''', r'''$\gamma$''']
        if dictate:
            lines=dictate[lobe]
            print(lines)
            lc = mc.LineCollection(lines, linewidths=2)
            ax.add_collection(lc)
            ax.autoscale()
            ax.margins(0.1)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Normalized spectral density')
        ax.set_title(lobe + " lobe - " + str(len(temp)) + " channels")
        for t, text in zip(coordinates, textes):
            ax.text(t, .085, text, fontsize=14)
        ax.set_xticks([0.5, 4, 8,13, 30, 80])
        ax.set_ylim(0,.10)
        ax.set_xlim(0.5, 80)
        
    plt.savefig(dataset + "_lobar_differences.svg", format="svg")
    if show:
        plt.show()
    else:
        plt.close()

# plot by lobe
def cutintervals(x):
    intervals=(0.5,0.75,1.25,1.75,2.25,3.25,3.75,4.25,5.25,6.25,6.75,7.75,8.25,9.25,10.25,11.75,13.25,15.25,17.25,20.25,24.25,31.75, 80)
    colBin, y = pd.cut(x,intervals,retbins=True,include_lowest=True)
    return colBin, y


def get_intervals(psd, colbin):
    dictionary = dict(zip(psd.columns, colbin))
    psd_intervals = psd.groupby(dictionary, axis=1).sum()
    psd_intervals.columns= psd_intervals.columns.astype(str, copy = False)
    return psd_intervals

def plot_intervals(psd_intervals, i ):
    fig, ax = plt.subplots(1, 1, figsize=(30,10))
    matplotlib.rcParams.update({'font.size': 10})
    if 'Lobe' in psd_intervals.columns:
        ax.bar(psd_intervals.columns[:-1],psd_intervals.iloc[i].drop(['Lobe']),width=0.9)
    else:
        ax.bar(psd_intervals.columns,psd_intervals.iloc[i],width=0.9)
    #ax.show()
    ax.set_title("mean share of power in a given bin")
    plt.savefig( "mean_power_share_in_intervals.png", format="png")
    return ax

def plot_ecdf(v1, v2, lobe, idxCol, l):
   from mlxtend.plotting import ecdf

   fig, ax = plt.subplots()
   ax, _, _ = ecdf(v1)
   # second ecdf
   #x2 = X[:, 1]
   ax, _, _ = ecdf(v2, ax=ax)
   ax.set_xlabel('Normalized spectral density in bin')
   ax.set_ylabel('ECDF')
   ax.set_title(lobe +  idxCol)
   ax.legend(["lobe", 'no peak'],loc='upper left')
   plt.savefig("images/" + lobe + "_" + str(l[0]) +"_" + str(l[1])  +  "_interval.svg", format="svg")

def return_signifcant(df, psd):
    import re
    import rpy2
    from rpy2.robjects.packages import importr
    rstats = importr('stats') 
    #psd_intervals = psd_intervals.astype(float)
    psd_intervals= get_intervals(psd, colbin)
    psd_intervals= psd_intervals.assign(Lobe =  psd["Lobe"])
    
    
    df_intervals = get_intervals(df, colbin)
    
    stat = pd.DataFrame(index=["stat_value", "pvalue"], columns=psd_intervals.columns)
    stat = stat.fillna(0) 
    # remove last column
    i = 0
    dictate =  dict()
    lobes = ['Occipital', 'Parietal',  'Frontal', 'Temporal']
    for lobe in lobes:
        temp = psd_intervals[psd_intervals.Lobe==lobe].drop(['Lobe'], axis=1)
        list_of_intervals = []
        for ( idxCol, v1 ), ( _, v2 ) in zip( temp.iteritems(), df_intervals.iteritems()):
           #print ( v1, v2, idxCol)
           # v1 is a column from lobes
           # v2 is  a column from no peak set
           
           t, pval = stats.ks_2samp(v1, v2)
           # details of r function https://rdrr.io/cran/dgof/man/ks.test.html
           # ther is no computation of True value only assymptotic aproximation
           v1 = rpy2.robjects.vectors.FloatVector(v1)
           v2 = rpy2.robjects.vectors.FloatVector(v2)
           htest = rstats.ks_test(v1, v2, alternative ="less", exact= False)
           htestlist=list(htest)
           t = htestlist[0][0]
           pval = htestlist[1][0]
           pval = pval * 4 * 22
           p = 0.01
           y = 0.08 
           if pval < p:
               p = re.compile('(?:\d+(?:\.\d*)?|\.\d+)')
               l = p.findall( idxCol)
               l = [float(i) for i in l]
               print(l)
               list_of_intervals.append([(l[0], y), (l[1], y)])
               print('list of intervals', *list_of_intervals, sep = ", ")
               print(lobe, idxCol, t, format(pval,  '.5f'))
               #plot_ecdf(v1, v2, lobe, idxCol, l)
    
    
    
               i = i + 1
        dictate[lobe]= list_of_intervals
        print('dictate', dictate)
    return dictate 
# Unsupervised classification of channels and no-peak set. 
# %%
# %% #for intracranial datata
dataset = " intracranial data"
f, psd = pesco.preprocess.prepare_psd()
#fig, ax = preprocess.iterate_old(psd, seed=3, start= 8, end=15)
psd_clust , smal = preprocess.plot_specific_clusterisation(psd.drop(['Region name', 'Lobe'], axis=1), f, 8, seed, dataset=dataset, ifall=False, nopeak=1 )
df , median = get_no_peak(psd_clust, smal)
colbin, y = cutintervals(f)
psd_mean= psd.median(axis=0)
psd_mean = pd.DataFrame(psd_mean).transpose()
psd_intervals_mean = get_intervals(psd_mean, colbin)

# %%


#%%
plot_intervals(psd_intervals_mean, 0)
dictate = return_signifcant(df, psd)
plot_lobes(psd_clust,psd, f,  smal,  dataset, dictate,show = True)

# %%

#from mlxtend.plotting import ecdf
#
#fig, ax = plt.subplots()
#ax, _, _ = ecdf(psd_mean.transpose())
## second ecdf
#ax.set_xlabel('Normalized spectral density in bin')
#ax.set_ylabel('ECDF')ecdf
# %%
#plt.close('all')

# %%
dataset = " reconstructed sources"

DATA_PATH = '/home/daniel/PhD/data/Mantini2018'
datafolder = '/home/daniel/PhD/notebooks/pesco-pipeline/data/preproc/'
raw_src , result= preprocess.load_sources(DATA_PATH)
#result.set_index('ch_name')
f, psd_source_clust  = preprocess.get_psd_mat(raw_src._data, raw_src.info["sfreq"],raw_src.info["ch_names"], name='power_ieeg.csv',  save_psd=False)
Y = psd_source_clust.join(result)
psd_clust , smal = preprocess.plot_specific_clusterisation(Y.drop(['Region name', 'Lobe','region_number', 'dataset'], axis=1), f, 6, seed, dataset=dataset,ifall=False, nopeak=4)
# %%
psd = Y.drop(['region_number', 'dataset'], axis=1)
df , median = get_no_peak(psd_clust, smal)
colbin, y = cutintervals(f)
psd_mean= psd.median(axis=0)
psd_mean = pd.DataFrame(psd_mean).transpose()
psd_intervals_mean = get_intervals(psd_mean, colbin)
plot_intervals(psd_intervals_mean, 0)

dictate = return_signifcant(df, psd)
plot_lobes(psd_clust,psd, f,  smal,  dataset, dictate,show = True)

#df = pd.DataFrame({'ChannelName' : []})
#df['ChannelName'] = raw_src.info["ch_names"]
    







# %% 
# Compare power in specifi intervals Build 
# 4th cluster
import cairosvg
import glob
for file in glob.glob("*.svg"):
    name = file.split('.svg')[0]
    cairosvg.svg2png(url=name+'.svg',write_to=name+'.png')
# define intervals
#intervals  

