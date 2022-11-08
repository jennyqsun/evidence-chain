# Created on 11/4/22 at 5:46 PM 

# Author: Jenny Sun
'''
this script convert what's from process_eeg.py in the following steps:
1. downsample

to sXXX_blockX_xxx_clean.pkl
with bad trials and bad chans index
'''

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
# from linepick import *
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from os import listdir
import pickle
from sklearn.decomposition import FastICA
import hdf5storage


# hnlpy repo pacakges
sys.path.append('/home/jenny/hnlpy/')
import timeop


#
datadir = '/ssd/rwchain-all/round2/'
behdir = datadir + 'rwchain-beh/'
eegdir = datadir + 'rwchain-eeg/'
subj = '106'

filedir = eegdir + 's' + subj + '/'
fnames = [filename for filename in os.listdir(filedir) if 'epoched' in filename]
fnames.sort()
print(len(fnames), 'files')

f = 2
eegfile = hdf5storage.loadmat(filedir + fnames[f])
print(filedir + fnames[f])
stimDur = eegfile['stimDur']
chanloc = eegfile['chanloc']
labels = eegfile['labels']
eeg = eegfile['data']
df = eegfile['df']
photocell = eegfile['photocell']
prestimDur = eegfile['prestimDur']
resp = eegfile['resp']
rt = eegfile['rt']
sr = eegfile['sr']
stimDur = eegfile['stimDur']

# step 1: let's downsample the data
eeg = eeg[:,::2,:]
sr = 1000
trialDur = eeg.shape[1]
print(trialDur, 'ms per trial')
nTrial = len(eeg)


# step 3: keep the data 1s prestim, 1s after rt, rest of nan
eegN = np.zeros_like(eeg)
for trial in range(nTrial):
    tend = int(1000+rt[trial]+1000)
    eegN[trial,:tend,:] = eeg[trial,:tend,:]
    eegN[trial, tend:, :] = np.nan




# step 3:  Identify bad trials by standard deviation
eegstd = np.nanstd(eegN, axis=1)
chanstd = np.nansum(eegstd, axis=0)
trialstd = np.nansum(eegstd, axis=1)

var_threshold = 2.5
ntrials = 50

badtrials_eeg = np.where(trialstd / np.median(trialstd) > var_threshold)[0]
goodtrials = np.setdiff1d(range(ntrials), badtrials_eeg)
badchan_eeg = np.where(chanstd/np.median(chanstd) > chan_threshold)[0]


# step 2: identify the abs bad channel that exceeds 100 mV
channelArtifact = np.nansum(np.abs(eegN)>100, axis=(0,-2))
# remove any channel if more than 20% of the time it's >100
badchan = np.where(channelArtifact> 0.2*eeg.shape[0]*eeg.shape[1])
print('badchan: ', badchan)
mask = np.ones(eeg.shape[2],bool)
mask[badchan] = False










# step 3:
# get the std 1s prestim + 1s after rt

chanstd = np.sum(eegstd[:, 0:nEEGchan], axis=0)
trialstd = np.sum(eegstd[:, 0:nEEGchan], axis=1)

if plot == "y":
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].plot(chanstd / np.median(chanstd), "ro")
    ax[0].set_title("Channel Variability")
    ax[0].set_xlabel("Channels")
    ax[0].set_ylabel("Normalized Standard Deviation")
    ax[0].grid()
    ax[1].plot(trialstd / np.median(trialstd), "bo")
    ax[1].set_xlabel("Trials")
    ax[1].set_ylabel("Normalized Standard Deviation")
    ax[1].set_title("Trial Variability")
    ax[1].grid()
    plt.show(block=False)