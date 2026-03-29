#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% [markdown]
# # Look for clusters
#
# ## to do 
# -  compute power again

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import matplotlib
import scipy
import pesco.pesco.preprocess as preprocess
#import pesco.utils as utils


# %%

def prepare_psd():
    #datafolder = '/home/daniel/PhD/notebooks/pesco-pipeline/data/preproc/'
    
    #df_ieeg = pd.read_csv(datafolder + "ieeg_raw.csv")
    #raw_ieeg = mne.io.read_raw_fif(datafolder +"ieeg_raw.fif",preload=True)
    
    
    mat = scipy.io.loadmat('/home/daniel/PhD/data/Frauscher2018/WakefulnessMatlabFile.mat')
    #utils.print_mat_nested(mat, indent=1, nkeys=30)
    patient= mat['Patient']
    ChannelName= [l.flatten()[0] for l in mat['ChannelName'].flatten()]
    #filtered_list = [ChannelName[i] for i in range(len(ChannelName)) if (patient==1)[i]]
    #specific_chans = raw_ieeg.copy().pick_channels(filtered_list)
    #from itertools import compress
    #temp = list(compress(ChannelName, (patient==1)))
    #temp=ChannelName[patient==1,]
    
    df = pd.DataFrame(patient, columns=['Patient'])
    df['ChannelName'] = mat['ChannelName']
    data = mat['Data'].T
    Fs=200.
    
    f, psd_df  = preprocess.get_psd_mat(data, Fs,ChannelName, name='power_ieeg.csv',  save_psd=False)
    return (f, psd_df)



def plot_single_psd(psd_df, i):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.semilogx(f, psd_df.iloc[i])
    ax.grid()
    #ax.plot(f, psd_df.iloc[1])
    ax.set_xlim(0.5, 80)
    ax.set_xticks([0.5, 4, 8,13, 30, 80])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #plt.yscale('log')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('PSD ')

def compute_clusters(psd_df, HowManyClusters, random_seed = 2):
    power = psd_df.values
    

    from sklearn.preprocessing import Normalizer
    from sklearn.cluster import KMeans
    from sklearn.pipeline import make_pipeline
    normalizer = Normalizer()
    np.random.seed(42)
    np.random.RandomState(3)
    
   
    np.random.seed(3)
    kmeans = KMeans(n_clusters=HowManyClusters, max_iter=300, n_init=100 , n_jobs=1, random_state=random_seed)
    pipeline = make_pipeline(normalizer, kmeans)
    pipeline.fit(power)
    #cluster labels
    cluster_labels = kmeans.labels_
    psd_df = psd_df.assign(clusters=cluster_labels)
    return psd_df

def plot_psd_clusters( psd_df, smal = [], nopeak= []):
    psd_medianas = psd_df.groupby('clusters').median()
    #HowManyClusters = len(psd_df["clusters"].unique())
    #plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    #plot means of the clusters
    for k in range(0, len(psd_medianas.index)):
        #srednia = np.mean(psd_df[cluster_labels==k]) 
        #srednia = srednia[:-1]
        mediana = psd_medianas.loc[ k]

        if k in smal:
            temp_label= 'cl. ' + str(k) + ' ('+ str(psd_df['clusters'].value_counts().loc[k]) +' el.)'
            ax.semilogx(f, mediana, linestyle=':', label=temp_label)
        elif k  == nopeak:
            temp_label= 'cl. ' + str(k) + ' ('+ str(psd_df['clusters'].value_counts().loc[k]) +' el.)'
            ax.semilogx(f, mediana, linestyle=':',linewidth=5.0, color="black", label=temp_label)
        else:
            temp_label= 'cl. ' + str(k) + ' ('+ str(psd_df['clusters'].value_counts().loc[k]) +' el.)'

            ax.semilogx(f, mediana, alpha=0.5, label=temp_label)

        #plt.xscale('log')
        #plt.xlim(0.5, 80)
        #xticks = [1, 2, 4, 8, 16, 32, 64]
        #ticklabels = ['1', '2', '4', '8', '16', '32', '64']
        #plt.xticks(xticks, ticklabels)
        
    #ax.legend(range(0,len(psd_medianas.index))  ) 
    ax.legend();
    ax.set_xticks([0.5, 4, 8,13, 30, 80])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid()
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Normalized spectral density')
    plt.title("Median power of different  PSDs clusters")
    plt.show()
    return (fig, ax)
    
    
def iterate_1(psd, seed, start, end):
    for HowManyClusters in range(start, end):
        print(HowManyClusters)
        psd_df =  compute_clusters(psd, HowManyClusters, seed)
        psd_medianas = psd_df.groupby('clusters').median()
        #psd_medianas = psd_medianas.sort(['cluster'])
        smal = []
        print() 
        for index, row in psd_medianas.iterrows():
          
            max_completion = psd_medianas.iloc[ psd_medianas.index != index, :].max()
            smaller= np.sum(np.less(row, max_completion))
            #print( "cluster ", index , " is smaller on  ", smaller , "frequencies")
            if smaller == 160:
                print() 
                print("cluster", index, "when we have" , HowManyClusters, "clusters")
                print() 
                smal.append(index)
        if smal:        
            plot_psd_clusters(psd_df, smal)
        print(psd_df['clusters'].value_counts())
        

def iterate_old(psd, seed, start, end):
    for HowManyClusters in range(start, end):
        print(HowManyClusters)
        psd_df =  compute_clusters(psd, HowManyClusters, seed)
        psd_medianas = psd_df.groupby('clusters').median()
        smal = []
        print() 
        for index, row in psd_medianas.iterrows():
#            temp = psd_medianas.copy.drop(psd_medianas.index[index])
          
            max_completion = psd_medianas.iloc[ psd_medianas.index != index, :].max()
            smaller= np.sum(np.less(row, max_completion))
            #print( "cluster ", index , " is smaller on  ", smaller , "frequencies")
            if smaller == 160:
                print() 
                print("cluster", index, "when we have" , HowManyClusters, "clusters")
                smal.append(index)
        if smal:        
           fig, ax =  plot_psd_clusters(psd_df, smal)
        print(psd_df['clusters'].value_counts())
    if smal: 
        return (fig, ax)
        
def iterate(psd, seed, start, end):
    for HowManyClusters in range(start, end):
        print(HowManyClusters)
        psd_df =  compute_clusters(psd, HowManyClusters, 3)
        psd_medianas = psd_df.groupby('clusters').median()
        psd_max = psd_df.groupby('clusters').max()
        #psd_medianas = psd_medianas.div(psd_medianas.sum(axis=1), axis=0)

        for index, row in psd_medianas.iterrows():
            temp = psd_max.copy().drop(psd_max.index[index])
            smal = []
            for index_in, row_in in temp.iterrows():
                smaller= np.sum(np.less(row, row_in))
                smal.append(smaller)
            print(np.count_nonzero(smal == 160)) 
            print(smal)
            if  np.count_nonzero(smal == 160) ==HowManyClusters:
                    print("aaa")
                    print(smal)
            #rint(smal)
            
def plot_one(psd, HowManyClusters, seed):
    psd_df =  compute_clusters(psd, HowManyClusters, seed)
    psd_medianas = psd_df.groupby('clusters').median()
    smal = []
    print() 
    for index, row in psd_medianas.iterrows():
#            temp = psd_medianas.copy.drop(psd_medianas.index[index])
      
        max_completion = psd_medianas.iloc[ psd_medianas.index != index, :].max()
        smaller= np.sum(np.less(row, max_completion))
        #print( "cluster ", index , " is smaller on  ", smaller , "frequencies")
        if smaller == 160:
            print() 
            print("cluster", index, "when we have" , HowManyClusters, "clusters")
            smal.append(index)
    for nopeak in range(0,HowManyClusters):        
       fig, ax =  plot_psd_clusters(psd_df, smal, nopeak)
       print(nopeak)
    print(psd_df['clusters'].value_counts())
# %%
f, psd = prepare_psd()
fig, ax = iterate_old(psd, seed=3, start= 8, end=15)

seed=3
start= 3
end=20

plot_one(psd, 9, seed)
# %%


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import plotly.plotly as py

import plotly 
plotly.tools.set_config_file(world_readable=True,                             sharing='public')
py.sign_in('danieltomasz', 'fNelk7vAfvJYBomaJXLF')

plot_url = py.plot_mpl(fig)
#
#import plotly.plotly as py
#
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#import plotly.io as pio
#import plotly.graph_objs as go
#
#plotly.offline.init_notebook_mode(connected=True)
#
#import plotly.graph_objs as go
#import pandas as pd
#
#
#from IPython.display import Image
#
#fig = go.FigureWidget()
#fig.add_scatter(y=[2, 4, 3, 2.5])
#iplot(fig)

#pio.write_json(fig, 'scatter.plotly')

# %%

# %%
#Unsupervised classification of channels and no-peak set
iterate(psd, seed=3, start= 3, end=20)

# %%
import mne
datafolder = '/home/daniel/PhD/notebooks/pesco-pipeline/data/preproc/'   
df_src = pd.read_csv(datafolder +"sources_raw.csv")
raw_src = mne.io.read_raw_fif(datafolder +"sources_raw.fif", preload=True)
X = raw_src._data
df = pd.DataFrame({'ChannelName' : []})
df['ChannelName'] = raw_src.info["ch_names"]
Fs=200.
    
f, psd_df_source  = preprocess.get_psd_mat(X, Fs,raw_src.info["ch_names"], name='power_ieeg.csv',  save_psd=False)

iterate_old(psd_df_source, seed=3, start= 8, end=15)
plot_one(psd_df_source, 8, seed)

# %%

#    
#psd_means= psd_df.groupby('clusters').mean()
#psd_maxs = psd_df.groupby('clusters').max()
#counts = psd_df.groupby('clusters').count()bbbbbbbbbb
   
   
    
#define  maximum
#We determined the presence of this group by requiring that its mean normalized spectrum be lower than the maximum among the other groups. 

# %%
def plot_max(psd_df):
    #maxy = np.zeros([HowManyClusters,160]) #HowManyClusters = numbers of clusters
    HowManyClusters = len(psd_df["clusters"].unique())
    for l in range(0, HowManyClusters):
        MeanClusterSpectrum = np.mean(psd_df["clusters"==l]) 
        MeanClusterSpectrum =  MeanClusterSpectrum[:-1]
        logicalmax=np.zeros([1,HowManyClusters])
        for k in range(0, HowManyClusters):
            # Maxima along the spectrum axis
            maxi = np.amax(psd_df["clusters"==k], 0)
            #last column is for  remembering label, we dont need it temporarily
            maxi = maxi[:-1]
            logicalsum= sum(np.less(MeanClusterSpectrum, maxi))
            if logicalsum == 160:
                logicalmax[0,k] = 1
        if  np.sum(logicalmax) == HowManyClusters:
            print(l)
        #print cluster number, if  averege of it is smaller in every bin than maximal values in other clusters



# %%
# estimate how good is clastering 
# more https://bl.ocks.org/rpgove/0060ff3b656618e9136b

def eblow(df, n):
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist, pdist
    import numpy as np


    kMeansVar = [KMeans(n_clusters=k).fit(df.values) for k in range(1, n)]
    centroids = [X.cluster_centers_ for X in kMeansVar]
    k_euclid = [cdist(df.values, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(df.values)**2)/df.values.shape[0]
    bss = tss - wcss
    plt.plot(bss)
    plt.show()

def eblow_psd(psd_df, n):
    from sklearn.cluster import KMeans

    psd_df = psd_df[0:160]
    
        
    sse = {}
    for k in range(1, n):
        
        kmeans = KMeans(n_clusters=k, max_iter=100).fit(psd_df)
        #data["clusters"] = kmeans.labels_
        #print(data["clusters"])
        sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()
#eblow(psd_df, 15)



# %%
import mne
datafolder = '/home/daniel/PhD/notebooks/pesco-pipeline/data/preproc/'   
df_src = pd.read_csv(datafolder +"sources_raw.csv")
raw_src = mne.io.read_raw_fif(datafolder +"sources_raw.fif", preload=True)
X = raw_src._data
df = pd.DataFrame({'ChannelName' : []})
df['ChannelName'] = raw_src.info["ch_names"]
Fs=200.
    
f, psd_df  = preprocess.get_psd_mat(X, Fs,raw_src.info["ch_names"], name='power_ieeg.csv',  save_psd=False)
