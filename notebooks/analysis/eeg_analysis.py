# Created on 1/23/23 at 4:53 PM 

# Author: Jenny Sun

import numpy as np
import matplotlib.pyplot as plt

# local pacakges
from model.get_behav_cleaned import *


projectdir = '/home/jenny/evidence-chain/'
datapath = '/ssd/rwchain-all/round2/rwchain-eeg/'


finalsubIDs = getIDs(datapath)



sub = finalsubIDs[0]
print(sub)

cond = 100
eegfile = datapath + sub
fnames, subid = getAllCondNames(datapath, finalsubIDs, cond)  # subid is unique value of each subject


# find the trials
data = np.empty((0,cond *30+1000+1000, 128))

# find the ones for 1 subject
fnames = [f for f in fnames if sub in f]

dataAll = np.empty((0,cond*30+2000, 128))
# stack the data
for i in fnames:
    subdict = loadsubjdict(i)
    data=subdict['data']
    dataAll = np.vstack((dataAll,data))

sr = subdict['sr']
# shortest trial
minRT = np.where(np.sum(np.isnan(data[:,:,0]), axis=0) >0)[0]

np.where(subdict['goodchans']==49)
np.where(subdict['goodchans']==49)
np.where(subdict['goodchans']==49)
goodchans = subdict['goodchans']
erp = np.nanmean(dataAll[:,500:2500,subdict['goodchans']], axis=0)
plt.figure()
plt.plot(np.arange(-500,1500),erp)
plt.show()

from scipy.fft import fft
plt.figure()
y = fft(erp, axis=0)
dur = 1.5
L = sr*dur # signal length
P2 = np.abs(y/L)
P1 = P2[0:int(L/2)]  # DC included
P1[1:] = 2 * P1[1:]

# define the frequency domain f
f = sr * np.arange(0, L/2) /L
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.title('%s'%sub + ' condition: %s ms'%cond +'\nSignal Duration: 1.5s')
plt.plot(f[1:31],P1[1:31])
plt.show()