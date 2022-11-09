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
import chanset

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

f = 0
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

eyeChans = [0, 1, 2]   # Fp1, Fpz, Fp2
nChans = eeg.shape[-1]
# step 1:  downsample the data
# check the sampling rate
print('sampling rate', sr)
trialDur = eeg.shape[1]
print(stimDur, 'ms per trial')
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
# common average referencing
eeg_combined = eeg_combined - np.tile(np.mean(eeg_combined, axis=1), (eeg_combined.shape[1],1)).T

# run ICA
# only using the good chans, using bad chan takes forever to converge or does not converge
print('running ICA...')
ICA = FastICA(n_components=sum(maskchan)-5, whiten=True, fun = 'cube', max_iter = 10000000)
S = ICA.fit_transform(eeg_combined[:,maskchan])
A = ICA.mixing_
W = ICA.components_
dubious_chans = np.unique(np.concatenate((np.array(eyeChans),np.where(~maskchan)[0])))

# recover the signal
recover= A @ S.T

fig0, ax0 = plt.subplots(4,1)
# original
ax0[0].plot(eeg_combined[1000:20000,maskchan])

#source
ax0[1].plot(S[1000:20000])

# let's remove the component
ax0[2].plot(recover[:,1000:20000].T)
fig0.suptitle('Recover signal')
fig0.show()

# compute correlations with eye channels and bad channels
# compute correlations with eye channels
ngoodchan = sum(maskchan)
nICAchan = sum(maskchan)-5
corrs = np.zeros((len(dubious_chans), nICAchan))
for j in range(nICAchan):
    for k in range(len(dubious_chans)):
        corrs[k, j] = np.corrcoef(S[:, j], eeg_combined[:, dubious_chans[k]])[0, 1]
# corrsmax = np.max(corrs, axis=0)
# plt.plot(corrsmax)

corr_threshold = 0.3
channelcorr =np.sum(np.abs(corrs)>corr_threshold,axis=0)
plt.plot(np.abs(corrs).T)
goodcomponents =  channelcorr==0
plt.ylabel('component correlations with bad chans')
plt.show()
print('good components: ', sum(goodcomponents))


# detect which components are not too correlated with eye channels
# goodcomponents = abs(corrsmax) < corr_threshold
chancomponents = np.zeros(nICAchan)
B = np.zeros(A.shape)
for j in range(nICAchan):
    B[:,j] = A[:,j]**2/np.sum(A[:,j]**2)
    chancomponents[j] = np.max(B[:,j])
    if (chancomponents[j] > 0.8):
        goodcomponents[j] = False

print('good components: ', sum(goodcomponents))




fig, ax = plt.subplots(sum(~goodcomponents)+3,1)
ax[0].plot(eeg_combined[:,0])
ax[1].plot(eeg_combined[:,1])
ax[2].plot(eeg_combined[:,2])
for i,cor in enumerate(np.where(~goodcomponents)[0]):
    ax[i+3].plot(S[1000:20*8000,cor])
fig.suptitle('corrleation between fp1 fp2 and components')
fig.show()


# recombine data without bad components.
AT = A.T
cleandata = S[:, goodcomponents] @ AT[goodcomponents, :]


# visualize it
ax0[3].plot(cleandata[1000:20000,:])
fig0.show()





# cleandata = signal.sosfiltfilt(sos,cleandata,axis=0,padtype='odd')


# re-epoch the clean data, and insert the bad channel exlucded from ICA
finaldata = np.zeros_like(eeg_trial[:,:,maskchan])
tstart = 0
for t_ in range(sum(masktrial)):
    trialRT = rt[masktrial][t_]
    tend = int(trialRT+1000+1000)
    finaldata[t_,:tend,:] = cleandata[tstart:tstart+tend, :]
    finaldata[t_, tend:, :] = np.nan
    tstart = tstart+tend


labels, positions, chans = chanset.chansets_neuroscan()


fig, ax = plt.subplots(2)
ax[0].plot(np.nanmean(eegN[:,1000:3000,chans], axis=0))

ax[1].plot(np.nanmean(finaldata[:,1000:3000,chans], axis=0))
fig.show()





