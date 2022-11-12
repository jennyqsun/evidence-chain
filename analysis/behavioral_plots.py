# Created on 11/11/22 at 2:57 PM 


'''
this script load all the behavioral data saved under
/eeg-behavior/combined/
and make plots and predictions
'''


import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import interactive
interactive(True)
import numpy as np
import os
import hdf5storage
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LinearRegression
import random
from sklearn.metrics import r2_score
import pickle
import matplotlib
from matplotlib.gridspec import GridSpec
from scipy.io import savemat
from bipolar import hotcold
import shutil
import os
from configparser import ConfigParser
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import time
import matplotlib.patches as mpatches
sys.path.append('/home/jenny/hnlpy/')
import timeop

# Author: Jenny Sun
# chop the data such that we have
# identify the frame where peak evidenc occur

def loadsubjdict(fnames):
    datadict = hdf5storage.loadmat(fnames)
    return datadict

cond = 50


datadir='/ssd/'
datadict = hdf5storage.loadmat()

cumsum = np.cumsum(sequenceall, axis=1)
rtadjusted = rtall-300
count = np.zeros_like(rtall)
val, m = divmod(rtadjusted, cond)
count[m==0] =val[m==0]   # it's exacly at the onset of the new stimulus,
                        # we say they made a decisiona fter seeing th previous ones
count[m>0] = val[m>0] + 1  # so, count it the last display that would count
                            # count =3 means they have seen display 0, 1, 2.

# now let's calculate first occurence of max evidene
maxval = []
maxind = []
sequence_stopAll = []
bound = []
boundind = []
nummax = []
for c in range(cumsum.shape[0]):
    sequence_stop = cumsum[c, :int(count[c])]   # for example, if rt adjusted i 708, for 250ms, one have seen 3 stimulus.
    sequence_stopAll.append(sequence_stop)
    bound.append(sequence_stop[-1])
    boundind.append(len(sequence_stop)-1)
    maxindtrial = np.where(np.abs(sequence_stop) ==np.max(abs(sequence_stop)))[0]
    maxind.append(maxindtrial)
    maxval.append(sequence_stop[maxindtrial])
    nummax.append(len(maxindtrial))

boundind = np.array(boundind)
maxfirst = [f[0] for f in maxind]
maxlast = [f[-1] for f in maxind]



ind = sidall ==1

fig, ax = plt.subplots()
offsetfirst = boundind-np.array(maxfirst)
sns.histplot(offsetfirst[ind], discrete=True, ax=ax)
plt.show()


fig, ax = plt.subplots()
offsetlast = boundind-np.array(maxlast)
sns.histplot(offsetlast[ind], discrete=True, ax=ax)
plt.show()



# fig, ax = plt.subplots()
# sns.barplot(offsetlast==0,offsetlast==1)
# s1,s2,s3 = sum(offsetlast==0), sum(offsetlast==1), sum(offsetlast>=2)


bound = np.array(bound)
# tangent, bondary plot for each subjects
import seaborn as sns
for sub in np.unique(sidall):
    plt.rcParams.update({'font.size': 17})
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    colors =sns.color_palette("deep")
    subind = sidall==sub
    subound = bound[subind]
    subresp = respall[subind]
    p = sns.histplot(subound[subresp>0],discrete=True, ax = ax[0],color = colors[3])
    p2 = sns.histplot(-subound[subresp<0],discrete=True,  ax = ax[1],color = colors[4])
    ax[0].set_xlabel('Evidence Towards O')
    ax[1].set_xlabel('Evidence Towards X')
    ax[0].set_xlim(-5,12)
    ax[1].set_xlim(-5,12)
    fig.tight_layout()
    fig.show()

    s0, s1,s2,s3 = sum(offsetfirst[subind]==0) / sum(subind),\
                   sum(offsetlast[subind]==0) / sum(subind), \
                   sum((offsetlast[subind]==1) | (offsetlast[subind]==2)) / sum(subind),\
               sum(offsetlast[subind]>=3) / sum(subind)
    fig, ax = plt.subplots()
    ax.bar((0,1,2),(s1,s2,s3) ,color = colors[0])
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['0', '1 or 2 Samples', '3 or More Samples'])
    fig.show()

subind = np.ones_like(sidall,dtype='bool')
s1,s2,s3 =    sum(offsetlast[subind]==0) / sum(subind), \
               sum((offsetlast[subind]==1) | (offsetlast[subind]==2)) / sum(subind),\
           sum(offsetlast[subind]>=3) / sum(subind)

s01, s02,s03 =sum(offsetfirst[subind]==0) / sum(subind),\
sum((offsetfirst[subind]==1) | (offsetfirst[subind]==2)) / sum(subind),\
           sum(offsetfirst[subind]>=3) / sum(subind)
fig, ax = plt.subplots()
ax.bar((0,3,6),(s1,s2,s3) ,color = colors[0])
ax.bar((1,4,7),(s01,s02,s03) ,color = colors[2])
ax.set_xticks([0,1,3,4,6,7])

ax.set_xticklabels(['0',' ', '1 or 2',' ', '3 or More',' '])
ax.set_xlabel('Number of offset samples between \n Peak of Evidence and Decision')
ax.set_ylabel('Percentage of Trials')
patch0 = mpatches.Patch(color=colors[0], label='Last Evidence Peak')
patch1 = mpatches.Patch(color=colors[2], label='First Evidence Peak')
fig.legend(handles=[patch0, patch1])
fig.tight_layout()
fig.show()
fig.savefig('offset.png')

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
colors = sns.color_palette("deep")
subound = bound[subind]
subresp = respall[subind]
p = sns.histplot(subound[subresp > 0], discrete=True, ax=ax[0], color=colors[3])
p2 = sns.histplot(-subound[subresp < 0], discrete=True, ax=ax[1], color=colors[4])
ax[0].set_xlabel('Evidence Towards O')
ax[1].set_xlabel('Evidence Towards X')
ax[0].set_xlim(-5, 12)
ax[1].set_xlim(-5, 12)
fig.tight_layout()
fig.show()
fig.savefig('boundall.png')

