# Created on 11/11/22 at 2:57 PM


'''
this script load all the behavioral data saved under
/eeg-behavior/combined/
and make plots and predictions
'''

#%%
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import interactive
interactive(True)
import numpy as np
import os
import hdf5storage
import random
from sklearn.metrics import r2_score
import pickle
import matplotlib
from matplotlib.gridspec import GridSpec
from scipy.io import savemat
import shutil
import os
from configparser import ConfigParser
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import time

from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import timeop


plt.rcParams.update({'font.size': 18})
colors = sns.color_palette("deep")

# local pacakges


import utils.movingAverage
from utils.movingAverage import *



#%%
# Author: Jenny Sun
# chop the data such that we have
# identify the frame where peak evidence occur

def loadsubjdict(fnames):
    datadict = hdf5storage.loadmat(fnames)
    return datadict

cond = 100

projectdir = '/home/jenny/evidence-chain/'
filename='/ssd/rwchain-all/round2/rwchain-beh/combined/all_' + str(cond) + '.mat'
datadict = hdf5storage.loadmat(filename)


masktrial = datadict['masktrial']
respall = datadict['resp']
sequenceall = datadict['sequence']
rtall = datadict['rt']
sidall = datadict['sidall']



cumsum = np.cumsum(sequenceall, axis=1)
rtadjusted = rtall-300
count = np.zeros_like(rtall)
val, m = divmod(rtadjusted, cond)
count[m==0] =val[m==0]   # it's exacly at the onset of the new stimulus,
                        # we say they made a decisiona after seeing th previous ones
count[m>0] = val[m>0] + 1  # so, count it the last display that would count
                            # count =3 means they have seen display 0, 1, 2.

count[count>=30] = 30

# calculate first occurence of max evidene
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
maxfirst = np.array([f[0] for f in maxind])
maxlast = np.array([f[-1] for f in maxind])
maxvallast = np.array([f[-1] for f in maxval])
maxvalfirst =  np.array([f[0] for f in maxval])
nummax = np.array([i.shape[0] for i in maxind])
bound = np.array(bound)
offsetfirst = boundind-np.array(maxfirst)
offsetlast = boundind-np.array(maxlast)



# all boundaries######
subind = np.ones_like(sidall,dtype='bool')

# # rt distribution
sns.set(style="whitegrid")
# fig, ax = plt.subplots(2,1, figsize= (6,6))
# ax = ax.flatten()
# # sns.histplot(rtall, ax= ax[0])
# # ax[0].set_title('RT distribution')
# sns.histplot(nummax,discrete=True, ax= ax[0],color = colors[2])
# ax[0].set_title('num of peak evidence samples \nprior to decision')
# # sns.histplot(maxvallast,discrete=True, ax= ax[2])
# sns.histplot(maxvalfirst,discrete=True, ax= ax[1], color = colors[2])
# ax[1].set_title('Distribution of Value of Last Peak Evidence ')
# fig.tight_layout()
# fig.show()




#%%
# all subject plot
#  Boundary at response
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
colors = sns.color_palette("Paired")
subound = bound[subind]
subresp = respall[subind]
p = sns.histplot(subound[subresp > 0], discrete=True, ax=ax[0], color=colors[1])
p2 = sns.histplot(-subound[subresp < 0], discrete=True, ax=ax[1], color=colors[5])
ax[0].set_xlabel('Evidence Towards O')
ax[1].set_xlabel('Evidence Towards X')
ax[0].set_xlim(-6, 12)
ax[1].set_xlim(-6, 12)
ax[0].axvline(0, linewidth = 5, color='black')
ax[1].axvline(0, linewidth = 5, color='black')

# boundary as a function of count
import pandas as pd
acc = np.zeros_like(rtall, dtype='bool')
acc[(bound>=0) & (respall >0)] = 1
acc[(bound<=0) & (respall <0)] = 1
df = pd.DataFrame({'boundary': np.abs(subound[acc]), 'count':count[acc]},columns=['boundary', 'count'])
out = df.groupby('count')
bmean = out.mean()
boundmean = np.array(bmean['boundary'])
n=1
bmeanbin = np.average(boundmean.reshape(-1, n), axis=1)
ax[2].plot(1+np.arange(0,len(bmeanbin)*n,n), bmeanbin)
ax[2].set_xticks(1+np.arange(0,len(bmeanbin)*n,4))
movingAverageWindow = 3
bmeanMovingAverage = getMovingAverage(bmeanbin,movingAverageWindow)
ax[2].plot(1+np.arange(0,len(bmeanbin),len(bmeanbin)/len(bmeanMovingAverage)), bmeanMovingAverage, linewidth = 10, color= 'green', alpha = 0.5,\
           label = 'Moving Average (window = %s)'%movingAverageWindow)
ax[2].set_xlabel('Number of samples seen')
ax[2].set_ylabel('Boundary')
fig.suptitle('Level of Evidence (Boundary) at Response (ISI: %s ms)'%str(cond))
ax[2].legend()
fig.show()
fig.tight_layout()
fig.savefig(projectdir +'plots/boundall_%s.png'%str(cond))



#%%
# percentage of trials that contain one, two or more peaks
# peak evidence analysis
fig, ax = plt.subplots(3,2,figsize=(9,12))
ax = ax.flatten()




# relationship between value of peak evidence and offset

s1,s2,s3 =    (nummax[subind]==1)[acc].sum() / sum(acc), \
               ((nummax[subind]==2) | (nummax[subind]==3))[acc].sum() / sum(acc),\
           (nummax[subind]>=4)[acc].sum() / sum(acc)


df = pd.DataFrame({'maxfirst': np.abs(offsetfirst[acc]), 'count':np.abs(maxvalfirst)[acc]},columns=['maxfirst', 'count'])


out = df.groupby('count')

# get how many trials per group
groupcount = out.count()
groupcount = groupcount.reset_index().to_numpy()

offsetman = out.mean()[0:12]
groupcount = groupcount[0:12,:]
offsetman = np.array(offsetman['maxfirst'])
n=1
offsetman = np.average(offsetman.reshape(-1, n), axis=1)

ax[0].plot(np.arange(1,len(offsetman)*n+1,n), offsetman, color = 'orange')
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_xlabel('Value of Peak Evidence')
ax[0].set_ylabel('Mean Number of Offset Samples')
ax[0].set_title('Offset as a function of \nMagnitude of Peak Evidence')

ax[2].bar(groupcount[:,0], groupcount[:,1],color = 'orange')
ax[2].set_xticks(groupcount[:,0])
ax[2].set_xlabel('Value of Peak Evidence')
ax[2].set_ylabel('Number of Trials')



# relationship between offset and timing of peak evidence
df = pd.DataFrame({'maxfirst': np.abs(offsetfirst[acc]), 'count':maxfirst[acc]},columns=['maxfirst', 'count'])

out = df.groupby('count')
offsetman = out.mean()
offsetman = np.array(offsetman['maxfirst'])


# get how many trials per group
groupcount = out.count()
groupcount = groupcount.reset_index().to_numpy()



n=1
offsetman = np.average(offsetman.reshape(-1, n), axis=1)
ax[1].plot(np.arange(0,len(offsetman)*n,n), offsetman, color='blue')
ax[1].set_xlabel('Timing of Peak Evidence Occurence')
ax[1].set_ylabel('Mean Number of Offset Samples')
ax[1].set_title('Offset as a function of \nTiming of Peak Evidence')

ax[3].bar(groupcount[:,0], groupcount[:,1],color = 'blue')
ax[3].set_xticks(groupcount[:,0][::4])
ax[3].set_xlabel('Timing of Peak Evidence')
ax[3].set_ylabel('Number of Trials')




ax[4].bar((0,1,2),(s1,s2,s3) ,color = colors[9])
ax[4].set_xticks([0,1,2])

ax[4].set_xticklabels(['1', '2 or 3','4 or more'])
ax[4].set_title('Number of Peak Evidence Seen Prior to Decision\n')
ax[4].set_ylabel('Percentage of Trials')
ax[4].set_xlabel('Number of Peak Evidence Seen')
ax[4].set_ylim(0,1)

s01, s02,s03 =sum(offsetfirst[subind]==0) / sum(subind),\
sum((offsetfirst[subind]==1) | (offsetfirst[subind]==2)) / sum(subind),\
           sum(offsetfirst[subind]>=3) / sum(subind)

ss01, ss02,ss03 =sum(offsetlast[subind]==0) / sum(subind),\
sum((offsetlast[subind]==1) | (offsetlast[subind]==2)) / sum(subind),\
           sum(offsetlast[subind]>=3) / sum(subind)

ax[5].bar((0,3,6),(s1,s2,s3) ,color = colors[9])
ax[5].bar((1,4,7),(s01,s02,s03) ,color = colors[4])
ax[5].set_xticks([0,1,3,4,6,7])
ax[5].set_xticklabels(['0',' ', '1 or 2',' ', '3 or More',' '])
ax[5].set_xlabel('Number of Offset samples between \n Peak of Evidence and Decision')
ax[5].set_ylabel('Percentage of Trials')
patch0 = mpatches.Patch(color=colors[9], label='Last Evidence Peak')
patch1 = mpatches.Patch(color=colors[4], label='First Evidence Peak')
ax[5].set_ylim(0,1)
fig.legend(handles=[patch0, patch1], bbox_to_anchor = (1,0.3))

ax[5].set_title('Offset Between Peak Evidence \nand Decision')
fig.suptitle('Peak Evidence Prior to Decision (ISI:%s ms)'%cond)
fig.tight_layout()
fig.show()
fig.savefig(projectdir +'plots/offset_%s.png'%str(cond))








#%%
# now let's do some simulation, and make similar plots

import matplotlib.lines as mlines
def plot_sorted(d, dur, ymin, ymax,log=True):
    # sorting ind
    rt_ind = np.argsort(d['rt_'])  # sort by rt from fastest to slowest
    boundary_ind = np.argsort(d['boundary'])  # sort from lowest boundary to highest
    maxvalue_ind = np.argsort(np.abs(d['maxvalue']))
    fig, ax = plt.subplots(1, figsize=(10, 7))

    for p in np.unique(np.abs(d['maxvalue'])):
        ind_p = np.abs(d['maxvalue']) == p
        # acc = d['direction'] == d['key_']
        # acc_p = acc[ind_p]
        maxind_p = d['maxind'][ind_p]
        count_p = d['count_'][ind_p]
        maxind_sort = np.argsort(maxind_p)
        maxind_p = maxind_p[maxind_sort]
        count_p = count_p[maxind_sort]
        #     print(sum(count_p>maxind_p) == len(count_p))
        # acc_p = acc_p[np.argsort(maxind_p)]
        yvec = np.linspace(p - 0.2, p + 0.7, len(maxind_p))
        a = ax.plot(maxind_p, yvec, '^')
        col = a[0].get_color()
        ax.plot(count_p + 0.5, yvec, 'o', color=col)
        # ax.grid(visible=True, axis='x', linestyle='-')

        ax.set_xticks(np.arange(0, 31))

        for i in range(0, len(count_p)):
            x1 = maxind_p[i]
            x2 = count_p[i] + 0.5
            ax.plot((x1, x2), (yvec[i], yvec[i]), ls='--', color=col)
        ax.set_yticks(np.unique(np.abs(d['maxvalue'])))
        label = [str(i) for i in np.arange(1, 31).tolist()] + ['  end \n of trial']
        ax.set_xticks(np.arange(0, 31))
        ax.set_xticklabels(label)
        ax.set_ylabel('Magnitude of Peak Evidence Upon Reponse')
        ax.set_xlabel('Number of Stimuli')

        trig = mlines.Line2D([], [], ls='none', color='black', marker='^',
                             markersize=15, label='The Display Where the Chain First Peaked Upon Response')
        circ = mlines.Line2D([], [], ls='none', color='black', marker='o',
                             markersize=15, label='The Display Where Subject Responded')

        ax.legend(handles=[trig, circ], bbox_to_anchor = (0.6,1))

        ax.set_ylim(ymin,ymax)
        if log:
            plt.yscale('log')
        ax.set_yticks(np.arange(ymin,ymax))
        ax.set_yticklabels(np.arange(ymin,ymax))
        plt.title('Stimulus Duration Per Display: %s ms' % dur)
    # fig.show()
    return fig, ax
d = dict()
# randomly choose 50 trials
seed =10
m= sum(acc)
samplesub = (sidall[acc]==2)

# samplesub = (sidall[acc]==2) | (sidall[acc]==1)| (sidall[acc]==0)
d = dict()
randnind = np.random.randint(0,sum(acc), sum(acc))
d['rt_'] = np.array(rtadjusted[acc])[samplesub]
d['boundary'] = np.array(np.abs(bound)[acc])[samplesub]
d['maxind'] = np.array(maxfirst[acc])[samplesub]
d['maxvalue']=np.array(np.abs(maxvalfirst[acc]))[samplesub]
d['count_'] = np.array(count[acc])[samplesub] -1


fig, ax = plot_sorted(d,cond,3,8, log=True)
fig.show()
fig.savefig(projectdir + '/plots/trialplots_reality_%s.png'%str(cond),
            bbox_inches = 'tight', pad_inches = 0.1)
#%%
############## DDM ##########

save =True
fig, ax = plt.subplots(1,3,figsize= (15,5))
seq = sequenceall

indlist =[]
val = []
maxind = []
maxvalue = []
bound_= 6
for i,j in enumerate(cumsum):
    ind = np.where(np.abs(j)>=bound_)[0]
    if len(ind) > 0:
        ind = ind[0]
    elif len(ind) == 1:
        pass
    else:
        ind = 29
    indlist.append(ind)
    val.append(np.abs((j[0:ind+1])[-1]))   # boundary value
    if i < 20:
        ax[0].plot(j[0:ind+1])
    maxind_t = np.argmax(np.abs((j[0:ind+1])))
    maxind.append(maxind_t)                       # index of max value
    maxvalue.append(((j[0:ind+1]))[maxind_t])
ax[0].set_xlabel('Number of Steps')
ax[0].axhline(bound_,color='red')
ax[0].axhline(-bound_,color='red',label ='boundary')
rt = [(i-1)*0.2+np.random.normal(0.3,0.05) for i in indlist]
sns.histplot(val, ax=ax[1],color='blue', discrete=True)
sns.histplot(rt, ax=ax[2],color='green')
ax[0].legend()
ax[0].set_title('Simulated Random Walk Trials')
ax[1].set_title('Simulated Histogram of Bound')
ax[1].set_xlabel('Boundary')
ax[2].set_xlabel('RT (s)')
ax[2].set_title('Simulated Histogram of RT')
fig.suptitle('Predicted Patterns according to DDM')
fig.tight_layout()
if save:
    fig.savefig(projectdir + 'plots/hypothesis_sim_DDM_%s.png'%str(cond))
fig.show()

samplesub = sidall[acc]==3
d['rt_'] = np.array(rt)[acc][samplesub]
d['boundary'] = np.array(val)[acc][samplesub]
d['maxind'] = np.array(maxind)[acc][samplesub]
d['maxvalue']=np.array(maxvalue)[acc][samplesub]
d['count_'] = np.array(indlist)[acc][samplesub]
fig, ax = plot_sorted(d,100,2,12,log =False)

fig.show()

#%%
######################### collappsing bound ########################
bound_= 7
fig, ax = plt.subplots(2,1,figsize= (5,10))

k=1.5
t = np.linspace(0,6,30)
# alpha = 4 * (1-k*(t/(t+0.1)))

alpha  = bound_-(1-np.exp(-(t/bound_)**k))*(0.6*bound_+6)

maxind = []
maxvalue = []

indlist =[]
val = []

h = []
for i,j in enumerate(cumsum):
    hit = np.abs(j)>=alpha
    h.append(any(hit))
    if any(hit):
        ind = np.where(hit)[0]
        if len(ind)>1:
            ind = ind[0]
        else:
            ind = ind[0]
    else:
        ind = 29
    indlist.append(ind)
    val.append(np.abs((j[0:ind + 1])[-1]))
    if i < 20:
        ax[0].plot(j[0:ind + 1])
    maxind_t = np.argmax(np.abs((j[0:ind+1])))
    maxind.append(maxind_t)
    maxvalue.append(((j[0:ind+1]))[maxind_t])
ax[0].plot(alpha,color='red')
ax[0].plot(-alpha,color='red',label='boundary')
ax[0].set_xlabel('Number of Steps')

rt = [(i-1)*0.2+np.random.normal(0.3,0.1) for i in indlist]
sns.histplot(val, ax=ax[1],color='blue',discrete=True)
# sns.histplot(rt, ax=ax[2],color='green')
ax[0].legend()
ax[0].set_title('Simulated Random Walk Trials')
ax[1].set_title('Simulated Histogram of Bound')
ax[1].set_xlabel('Boundary')
# ax[2].set_xlabel('RT (s)')
# ax[2].set_title('Simulated Histogram of RT')
fig.suptitle('Predicted Patterns according to cDDM')
fig.tight_layout()
fig.show()
fig.savefig(projectdir + 'plots/hypothesis_sim_cDDM.png')

# randomly choose 50 trials

samplesub = sidall[acc]==3
d['rt_'] = np.array(rt)[acc][samplesub]
d['boundary'] = np.array(val)[acc][samplesub]
d['maxind'] = np.array(maxind)[acc][samplesub]
d['maxvalue']=np.array(maxvalue)[acc][samplesub]
d['count_'] = np.array(indlist)[acc][samplesub]
fig, ax = plot_sorted(d,100,3,9,log =True)
fig.show()
fig.savefig(projectdir + 'plots/hypothesis_sim_cDDM_trial.png')


#%%
# ###########   Collasping bound vs. Reality   #####################


# all subject plot
#  Boundary at response
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
colors = sns.color_palette("Paired")
subound = bound[subind]
subresp = respall[subind]
# boundary as a function of count
import pandas as pd
acc = np.zeros_like(rtall, dtype='bool')
acc[(bound>=0) & (respall >0)] = 1
acc[(bound<=0) & (respall <0)] = 1
df = pd.DataFrame({'boundary': np.abs(subound[acc]), 'count':count[acc]},columns=['boundary', 'count'])
out = df.groupby('count')
bmean = out.mean()
boundmean = np.array(bmean['boundary'])
n=1
bmeanbin = np.average(boundmean.reshape(-1, n), axis=1)
ax.plot(1+np.arange(0,len(bmeanbin)*n,n), bmeanbin,color = 'grey')
ax.set_xticks(1+np.arange(0,len(bmeanbin)*n,4))
movingAverageWindow = 3
bmeanMovingAverage = getMovingAverage(bmeanbin,movingAverageWindow)
ax.plot(1+np.arange(0,len(bmeanbin),len(bmeanbin)/len(bmeanMovingAverage)), bmeanMovingAverage, linewidth = 7, color= 'purple', alpha = 0.8,\
           label = 'Human Data (Moving Average window = %s)'%movingAverageWindow)
ax.set_xlabel('Number of samples seen')
ax.set_ylabel('Boundary Mean')
fig.suptitle('Level of Evidence (Boundary) at Response (ISI: %s ms)'%str(cond))
ax.legend()



# collasping bound
bound_= 7

k=1.5
t = np.linspace(0,6,30)
# alpha = 4 * (1-k*(t/(t+0.1)))

alpha  = bound_-(1-np.exp(-(t/bound_)**k))*(0.6*bound_+6)

maxind = []
maxvalue = []

indlist =[]
val = []

h = []
for i,j in enumerate(cumsum):
    hit = np.abs(j)>=alpha
    h.append(any(hit))
    if any(hit):
        ind = np.where(hit)[0]
        if len(ind)>1:
            ind = ind[0]
        else:
            ind = ind[0]
    else:
        ind = 29
    indlist.append(ind)
    val.append(np.abs((j[0:ind + 1])[-1]))
    maxind_t = np.argmax(np.abs((j[0:ind+1])))
    maxind.append(maxind_t)
    maxvalue.append(((j[0:ind+1]))[maxind_t])



valB = np.array(val)[acc]
countB = np.array(indlist)[acc] +1
df_bound = pd.DataFrame({'boundary': valB, 'count':countB},columns=['boundary', 'count'])
out = df_bound.groupby('count')
bmean = out.mean()
boundmean = np.array(bmean['boundary'])
n=1
bmeanbin = np.average(boundmean.reshape(-1, n), axis=1)
xlabel = out.count().index
ax.plot(xlabel, bmeanbin,linewidth = 7, label = 'Simulated Data',color= 'orange')
# ax.set_xticks(1+np.arange(0,len(bmeanbin)*n,4))
# movingAverageWindow = 3
# bmeanMovingAverage = getMovingAverage(bmeanbin,movingAverageWindow)
# ax.plot(1+np.arange(0,len(bmeanbin),len(bmeanbin)/len(bmeanMovingAverage)), bmeanMovingAverage, linewidth = 7, color= 'green', alpha = 0.5,\
#            label = 'Human Data (Moving Average w/ window = %s)'%movingAverageWindow)
ax.set_xlabel('Number of samples seen')
ax.set_ylabel('Boundary Mean')
fig.suptitle('Level of Evidence (Boundary) \nat Response (ISI: %s ms)'%str(cond))
ax.legend(bbox_to_anchor=(0.1, 0., 0.5, 0.5))
fig.show()
fig.tight_layout()

fig.savefig('/home/jenny/Downloads/' + 'humanVScDDM.png')




#%%
# all subject plot
#  Boundary at response
fig, ax = plt.subplots(2, 1, figsize=(5, 5))
colors = sns.color_palette("Paired")
subound = bound[subind]
subresp = respall[subind]
p = sns.histplot(subound[subresp > 0], discrete=True, ax=ax[0], color=colors[1])
p2 = sns.histplot(subound[subresp < 0], discrete=True, ax=ax[0], color=colors[5])
ax[0].set_xlabel('Evidence Towards X or O')
ax[0].set_xlim(-12, 12)
ax[0].axvline(0, linewidth = 5, color='black')
ax[0].axvline(0, linewidth = 5, color='black')






# relationship between value of peak evidence and offset

s1,s2,s3 =    (nummax[subind]==1)[acc].sum() / sum(acc), \
               ((nummax[subind]==2) | (nummax[subind]==3))[acc].sum() / sum(acc),\
           (nummax[subind]>=4)[acc].sum() / sum(acc)

tt = ~(np.abs(offsetfirst[acc]) >=27)
df = pd.DataFrame({'maxfirst': np.abs(offsetfirst[acc])[tt], 'count':np.abs(maxvalfirst)[acc][tt]},columns=['maxfirst', 'count'])


out = df.groupby('count')

# get how many trials per group
groupcount = out.count()
groupcount = groupcount.reset_index().to_numpy()

offsetman = out.mean()[0:10]
groupcount = groupcount[0:10,:]
offsetman = np.array(offsetman['maxfirst'])
n=1
offsetman = np.average(offsetman.reshape(-1, n), axis=1)

ax[1].plot(np.arange(1,len(offsetman)*n+1,n), offsetman, color = 'black')
ax[1].bar(np.arange(1,len(offsetman)*n+1,n),offsetman, color = 'teal')

ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set_xlabel('Value of Peak Evidence')
ax[1].set_ylabel('Mean Number of Offset Samples')
ax[1].set_title('Offset as a function of Magnitude of Peak Evidence')
ax[1].set_xticks(np.arange(1,len(offsetman)*n+1,n)*n,1)

fig.tight_layout()
fig.show()
fig.savefig('/home/jenny/Downloads/' + 'boundary_offset.png')




#%%
fig, ax = plt.subplots(2, 1, figsize=(5, 5))
colors = sns.color_palette("Paired")
subound = bound[subind]
subresp = respall[subind]
p = sns.histplot(subound[subresp > 0], discrete=True, ax=ax[0], color=colors[1])
p2 = sns.histplot(subound[subresp < 0], discrete=True, ax=ax[0], color=colors[5])
ax[0].set_xlabel('Evidence Towards X or O')
ax[0].set_xlim(-12, 12)
ax[0].axvline(0, linewidth = 5, color='black')
ax[0].axvline(0, linewidth = 5, color='black')






# relationship between value of peak evidence and offset

s1,s2,s3 =    (nummax[subind]==1)[acc].sum() / sum(acc), \
               ((nummax[subind]==2) | (nummax[subind]==3))[acc].sum() / sum(acc),\
           (nummax[subind]>=4)[acc].sum() / sum(acc)

tt = ~(np.abs(offsetfirst[acc]) >=27)
df = pd.DataFrame({'maxfirst': np.abs(offsetfirst[acc])[tt], 'count':np.abs(maxvalfirst)[acc][tt]},columns=['maxfirst', 'count'])


out = df.groupby('count')

# get how many trials per group
groupcount = out.count()
groupcount = groupcount.reset_index().to_numpy()

offsetman = out.mean()[0:10]
groupcount = groupcount[0:10,:]
offsetman = np.array(offsetman['maxfirst'])
n=1
offsetman = np.average(offsetman.reshape(-1, n), axis=1)

ax[1].plot(np.arange(1,len(offsetman)*n+1,n), offsetman, color = 'black')
ax[1].bar(np.arange(1,len(offsetman)*n+1,n),offsetman, color = 'teal')

ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set_xlabel('Value of Peak Evidence')
ax[1].set_ylabel('Mean Number of Offset Samples')
ax[1].set_title('Offset as a function of Magnitude of Peak Evidence')
ax[1].set_xticks(np.arange(1,len(offsetman)*n+1,n)*n,1)

fig.tight_layout()
fig.show()













#%% for the grant

bound_= 7
fig, ax = plt.subplots(2,1,figsize= (5.2,5))

k=1.5
t = np.linspace(0,6,30)
# alpha = 4 * (1-k*(t/(t+0.1)))

alpha  = bound_-(1-np.exp(-(t/bound_)**k))*(0.6*bound_+6)

maxind = []
maxvalue = []

indlist =[]
val = []

h = []
for i,j in enumerate(cumsum):
    hit = np.abs(j)>=alpha
    h.append(any(hit))
    if any(hit):
        ind = np.where(hit)[0]
        if len(ind)>1:
            ind = ind[0]
        else:
            ind = ind[0]
    else:
        ind = 29
    indlist.append(ind)
    val.append(np.abs((j[0:ind + 1])[-1]))
    if i < 20:
        ax[0].plot(j[0:ind + 1])
    maxind_t = np.argmax(np.abs((j[0:ind+1])))
    maxind.append(maxind_t)
    maxvalue.append(((j[0:ind+1]))[maxind_t])
ax[0].plot(alpha,color='red')
ax[0].plot(-alpha,color='red',label='boundary')
ax[0].set_xlabel('Number of Steps')
ax[0].set_ylabel('Evidence')

rt = [(i-1)*0.2+np.random.normal(0.3,0.1) for i in indlist]
# sns.histplot(rt, ax=ax[2],color='green')
ax[0].legend()
# ax.set_title('Simulated Random Walk Trials')

# ax[2].set_xlabel('RT (s)')
# ax[2].set_title('Simulated Histogram of RT')
# fig.suptitle('Predicted Patterns according to cDDM')
# fig.savefig('/home/jenny/Downloads/' + 'hypothesis_sim_cDDM_flat.png')


# all subject plot
#  Boundary at response
# fig, ax = plt.subplots(1, 1, figsize=(5.5, 3))
colors = sns.color_palette("Paired")
subound = bound[subind]
subresp = respall[subind]
# boundary as a function of count
import pandas as pd
acc = np.zeros_like(rtall, dtype='bool')
acc[(bound>=0) & (respall >0)] = 1
acc[(bound<=0) & (respall <0)] = 1
df = pd.DataFrame({'boundary': np.abs(subound[acc]), 'count':count[acc]},columns=['boundary', 'count'])
out = df.groupby('count')
bmean = out.mean()
boundmean = np.array(bmean['boundary'])
n=1
bmeanbin = np.average(boundmean.reshape(-1, n), axis=1)
ax[1].plot(1+np.arange(0,len(bmeanbin)*n,n), bmeanbin,color = 'grey')
ax[1].set_xticks(1+np.arange(0,len(bmeanbin)*n,4))
movingAverageWindow = 3
bmeanMovingAverage = getMovingAverage(bmeanbin,movingAverageWindow)
ax[1].plot(1+np.arange(0,len(bmeanbin),len(bmeanbin)/len(bmeanMovingAverage)), bmeanMovingAverage, linewidth = 7, color= 'purple', alpha = 0.8,\
           label = 'Human Data')
ax[1].set_xlabel('Number of samples seen')
ax[1].set_ylabel('Boundary Mean')
# fig.suptitle('Level of Evidence (Boundary) at Response (ISI: %s ms)'%str(cond))
# ax[1].legend()



# collasping bound
bound_= 7

k=1.5
t = np.linspace(0,6,30)
# alpha = 4 * (1-k*(t/(t+0.1)))

alpha  = bound_-(1-np.exp(-(t/bound_)**k))*(0.6*bound_+6)

maxind = []
maxvalue = []

indlist =[]
val = []

h = []
for i,j in enumerate(cumsum):
    hit = np.abs(j)>=alpha
    h.append(any(hit))
    if any(hit):
        ind = np.where(hit)[0]
        if len(ind)>1:
            ind = ind[0]
        else:
            ind = ind[0]
    else:
        ind = 29
    indlist.append(ind)
    val.append(np.abs((j[0:ind + 1])[-1]))
    maxind_t = np.argmax(np.abs((j[0:ind+1])))
    maxind.append(maxind_t)
    maxvalue.append(((j[0:ind+1]))[maxind_t])



valB = np.array(val)[acc]
countB = np.array(indlist)[acc] +1
df_bound = pd.DataFrame({'boundary': valB, 'count':countB},columns=['boundary', 'count'])
out = df_bound.groupby('count')
bmean = out.mean()
boundmean = np.array(bmean['boundary'])
n=1
bmeanbin = np.average(boundmean.reshape(-1, n), axis=1)
xlabel = out.count().index
ax[1].plot(xlabel, bmeanbin,linewidth = 7, label = 'Simulated Data',color= 'orange')
# ax.set_xticks(1+np.arange(0,len(bmeanbin)*n,4))
# movingAverageWindow = 3
# bmeanMovingAverage = getMovingAverage(bmeanbin,movingAverageWindow)
# ax.plot(1+np.arange(0,len(bmeanbin),len(bmeanbin)/len(bmeanMovingAverage)), bmeanMovingAverage, linewidth = 7, color= 'green', alpha = 0.5,\
#            label = 'Human Data (Moving Average w/ window = %s)'%movingAverageWindow)
ax[1].set_xlabel('Number of Samples Seen')
ax[1].set_ylabel('Boundary Mean')
# fig.suptitle('Level of Evidence (Boundary) \nat Response (ISI: %s ms)'%str(cond))
ax[1].legend(bbox_to_anchor=(0.1, 0.2, 0.6, 0.1))

fig.tight_layout()
fig.show()
fig.savefig('/home/jenny/Downloads/' + 'cDDM_humanVScDDM.png')


#%% rt from peak evidence

fig, ax = plt.subplots(1,2,figsize= (10,5))
df_PeakToRT = pd.DataFrame({'peakLevel': maxvalfirst[acc],
             'PeakToRT':rtall[acc]- (maxfirst[acc] + 1) * cond},columns=['peakLevel','PeakToRT'])

out = df_PeakToRT.groupby('peakLevel')
xlabel = out.count().index
# ax.scatter(xlabel, out.mean())
errorbars = (out.std().to_numpy()).squeeze() / np.sqrt((out.size().to_numpy()).squeeze())
ax[0].errorbar(xlabel.to_numpy(),out.mean().to_numpy(), np.nan_to_num(errorbars),fmt='-o')
ax[0].set_xticks(xlabel)
# ax[0].set_xlim(-8.5,8.5)
ax[0].set_ylabel('Mean RT from First Peak Evidence')
ax[0].set_xlabel('Level of Peak Evidence')

ax[1].bar(xlabel.to_numpy(), out.size().to_numpy())
ax[1].set_xticks(xlabel)
ax[1].set_xlim(-8.5,8.5)
ax[1].set_ylabel('Number of Trials')
ax[1].set_xlabel('Level of Peak Evidence')
fig.tight_layout()
fig.show()


# plt.figure()
# for level in range(-8,8):
#     ind =  maxvalfirst[acc] == level
#     plt.scatter(level, np.std(df_PeakToRT['PeakToRT'][ind]))

# plt.figure()
# for level in range(1,7):
#     ind =  maxvalfirst[acc] == level
#     plt.violinplot((df_PeakToRT['PeakToRT'][ind]),[level],showmedians=True)

# plt.ylim(0,2000)




#%%

fig, ax = plt.subplots(1,1,figsize= (10,4))
for i, k in enumerate(out.groups.keys()):
    if np.abs(k) <=5 and np.abs(k)>1:
        v = ax.violinplot(df_PeakToRT['PeakToRT'][out.groups[k]], positions = [np.abs(k)],showmedians=True)
ax.set_xlabel('Level of Evidence')
ax.set_ylabel('First Peak to RT distribution')
fig.tight_layout()
fig.show()


fig, ax = plt.subplots(2, 4, figsize=(10, 4))
ax = ax.reshape(-1)
count = 0
for i, k in enumerate(out.groups.keys()):

        v = ax[count].hist(out.groups[k])
        ax[count].set_title('Peak Level: %s'%k)
        count +=1


count = -1
for i, k in enumerate(out.groups.keys()):
    if (k) >= -5 and (k) < -1:
        v = ax[count].hist(out.groups[k])
        ax[count].set_title('Peak Level: %s'%k)
        count -=1
fig.tight_layout()
fig.show()


#%%

# ax.scatter(xlabel, out.mean())
errorbars = (out.std().to_numpy()).squeeze() / np.sqrt((out.size().to_numpy()).squeeze())
ax[0].errorbar(xlabel.to_numpy(),out.mean().to_numpy(), np.nan_to_num(errorbars),fmt='-o')
ax[0].set_xticks(xlabel)
ax[0].set_xlim(-8.5,8.5)
ax[0].set_ylabel('Mean RT from Last Peak Evidence')
ax[0].set_xlabel('Level of Peak Evidence')

ax[1].bar(xlabel.to_numpy(), out.size().to_numpy())
ax[1].set_xticks(xlabel)
ax[1].set_xlim(-8.5,8.5)
ax[1].set_ylabel('Number of Trials')
ax[1].set_xlabel('Level of Peak Evidence')
fig.tight_layout()
fig.show()

fig, ax = plt.subplots(1,1,figsize= (10,4))
for i, k in enumerate(out.groups.keys()):
    if np.abs(k) <=5 and np.abs(k)>1:
        v = ax.violinplot(out.groups[k], positions = [np.abs(k)])
ax.set_xlabel('Level of Evidence')
ax.set_ylabel('Last Peak to RT distribution')
fig.tight_layout()
fig.show()


fig, ax = plt.subplots(2, 4, figsize=(10, 4))
ax = ax.reshape(-1)
count = 0
for i, k in enumerate(out.groups.keys()):
    if (k) <= 5 and (k) > 1:
        v = ax[count].hist(out.groups[k])
        ax[count].set_title('Peak Level: %s'%k)
        count +=1

count = -1
for i, k in enumerate(out.groups.keys()):
    if (k) >= -5 and (k) < -1:
        v = ax[count].hist(out.groups[k])
        ax[count].set_title('Peak Level: %s'%k)
        count -=1
fig.tight_layout()
fig.show()
# %%
