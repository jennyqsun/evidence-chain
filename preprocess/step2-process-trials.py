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
subj = '103'

filedir = eegdir + 's' + subj + '/'
fnames = [filename for filename in os.listdir(filedir) if 'epoched' in filename]
fnames.sort()
print(len(fnames), 'files')

f = 3
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

nChans = eeg.shape[-1]
# step 1:  downsample the data
# check the sampling rate
print('sampling rate', sr)
trialDur = eeg.shape[1]
print(trialDur, 'ms per trial')
nTrial = len(eeg)


# step 2: keep the data 1s prestim, 1s after rt, rest is nan
eegN = np.zeros_like(eeg)
for trial in range(nTrial):
    tend = int(1000+rt[trial]+1000)
    eegN[trial,:tend,:] = eeg[trial,:tend,:]
    eegN[trial, tend:, :] = np.nan

# step 3: identify any obvious bad channel that exceeds 100 mV more th an 20% of the time
channelArtifact = np.sum(np.abs(eegN)>100, axis=(0,1))
# remove any channel if more than 20% of the time it's >100
badchan = np.where(channelArtifact> 0.2*np.sum(~np.isnan(eegN[:,:,0])))
print('badchan identification: ', badchan)
maskchan, masktrial = np.ones(eegN.shape[2],bool), np.ones(eegN.shape[0],bool)
maskchan[badchan] = False


# step 4:  Identify bad trials by standard deviation
eegstd = np.nanstd(eegN, axis=1)   # get the std over time for each trial, each channel
chanstd = np.nansum(eegstd, axis=0)  # get the std of each channel over trials
trialstd = np.nansum(eegstd, axis=1)  # get the std of each trial over channel


# threshhold trials by standard deviation criteria and remove them.
# therhold channels by standard devaition criteria

var_threshold = 2.5
chan_threshold = 2.5
ntrials = 50

badtrials_eeg = np.where(trialstd / np.median(trialstd) > var_threshold)[0]
goodtrials = np.setdiff1d(range(ntrials), badtrials_eeg)
badchan_eeg = np.where(chanstd/np.median(chanstd) > chan_threshold)[0]
print('bad trials', badtrials_eeg)
print('bad chans', badchan_eeg)

masktrial[badtrials_eeg] = False
maskchan[badchan_eeg] = False

# step 5: mask the trials where rt is invalid or <300
nortTrials = np.where((rt == -999) | (rt <=300))
masktrial[nortTrials] = False

#
# if the epoch before maxRT is larger then a number, reject
threshold = 200
badind = np.where(np.max(np.abs(eegN[:,1000:int(np.median(rt+1000)),maskchan]),axis=(-1,-2)) > threshold)[0]
masktrial[badind] = False

print('good trials: ', sum(masktrial))
print('good chans: ', sum(maskchan))

# Step 6: RUN ICA analysis to find and remove components correlated to eye blinks
# run ICA
eegN = eegN.astype(float)
eeg_trial = eegN[masktrial,:,:]
tTotal = np.sum(~np.isnan(eeg_trial[:,:,0]))

# concatenant the trials without NaN values
eeg_combined=np.zeros((0,eeg_trial.shape[-1]))
for i in range(sum(masktrial)):
    trialRT = rt[masktrial][i]
    trialEEG = eeg_trial[i,:,:]
    trialEEG = trialEEG[0: sum(~np.isnan(eeg_trial[i,:,0])),:]  # get the nanvalue
    eeg_combined = np.vstack((eeg_combined, trialEEG))
# remove the mean before ICA
eeg_combined = eeg_combined - np.tile(np.mean(eeg_combined, axis=1), (eeg_combined.shape[1],1)).T

# run ICA
ICA = FastICA(n_components=nChans, whiten=True)
S = ICA.fit_transform(eeg_combined)
A = ICA.mixing_
W = ICA.components_
dubious_chans = np.unique(np.concatenate((np.array(eyechans),badchan_eeg)))












# trial rejection by visual inpsection
mask_trial = np.ones(eegN.shape[0],bool)
mask_trial[rt==-999] = False
ind = np.where(np.abs(eeg) >=100)

fig, ax = plt.subplots(5,10,figsize=(35,20))
for i, j in enumerate(ax.flat):
    j.plot(eegN[:,:,mask][i,:,0:])
    j.axvline(rt[i]+1000)
    j.set_title(i)
fig.suptitle('whole EEG trial -1200 to end with masked channels')
fig.tight_layout()
fig.show()









