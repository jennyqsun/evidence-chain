'''Step 1'''

#%%
import curryreader as cr
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.signal as signal
import matplotlib as mpl
from matplotlib import interactive
interactive(True)
import numpy as np
import pandas as pd
import os
from os import listdir
import matplotlib.pyplot as plt
import pickle
import pyarrow.feather as feather

# global plot setting
plt.rcParams.update({'font.size': 17})
mpl.rcParams['lines.markersize'] = 8
import hdf5storage


# hnlpy repo pacakges
sys.path.append('/home/jenny/hnlpy/')
import timeop

'''
author: jenny
this script convert the .cdt file 
and convert them into blocks
'''


#
datadir = '/ssd/rwchain-all/round2/'
behdir = datadir + 'rwchain-beh/'
eegdir = datadir + 'rwchain-eeg/'
subj = '102'
ses = 1
# read and concatenance .csv file of the behavioral data
onlyfile_s1 = [f for f in listdir(behdir) if subj in f and 'csv' in f and 'ses1' in f and '#' not in f]
onlyfile_s2 = [f for f in listdir(behdir) if subj in f and 'csv' in f and 'ses2' in f and '#' not in f]

onlyfile_s1.sort()
onlyfile_s2.sort()
print(onlyfile_s1)
print(onlyfile_s2)
print(len(onlyfile_s1), 'files found session 1')
print(len(onlyfile_s2), 'files found session 2')

#%%
# concat trials
df = []
df1 = []
df2 =[]
for f in onlyfile_s1:
    blocknum = f[15]
    blockdf = pd.read_csv(behdir + f)
    blockdf['block'] = blocknum
    df.append(blockdf)
    df1.append(blockdf)
for f in onlyfile_s2:
    blocknum = f[15]
    blockdf = pd.read_csv(behdir + f)
    blockdf['block'] = blocknum
    df.append(blockdf)
    df2.append(blockdf)
df = pd.concat(df)
df1 = pd.concat(df1)
df2 = pd.concat(df2)


# get the unique condition displays
# condList=[]
# for d in df:
#     condList.append(d['stimDur'].unique())


def loadPKL(filename):
    myfile = open(filename, 'rb')
    f = pickle.load(myfile)
    return f


output = feather.read_feather(behdir + 'ftr-files/' + subj + '_allCond.ftr')


# use curry to read data
# currydata = cr.read()

def getEEG(fname):
    currydata = cr.read(fname, plotdata = 0, verbosity = 1)
    eegdata = currydata['data'][:,0:128]
    sr = currydata['info']['samplingfreq']
    photocell=currydata['data'][:,132:134]
    labels = currydata['labels']
    chanloc = currydata['sensorpos']
    print('sr: ', sr)
    return currydata, eegdata,sr,photocell,labels,chanloc


# bandpass fitler
def filter(datain,sr, low,low_stop, high, high_stop,gs):
    sos, w, h = timeop.makefiltersos(sr, low, low_stop)
    erpfilt_hi = signal.sosfiltfilt(sos,datain,axis=0,padtype='odd')
    sos2, w2, h2 = timeop.makefiltersos(sr, high, high_stop, gs=gs)
    eegdata = signal.sosfiltfilt(sos2,erpfilt_hi,axis=0,padtype='odd')
    return eegdata


def getAllArrays(df_1):
    ''' this function preprocess some empty rt collected
    :param df_1: behvairal dataframe
    :return: dictionary, contains subsets of columns from '''
    allData = {}
    rt = np.array(df_1['time'])
    rt [rt =='[]'] = -999
    rt = rt.astype('float')
    count = np.array(df_1['count'])

    # clean the data
    key = []
    sequence = []
    stimDur = []

    for index, row in df_1.iterrows():
        if row['key'] == "[5]":
            k = 1
        else:
            k = 0
        key.append(k)
        seq = row['sequence'].split(".")
        l = []
        for i in seq:
            i = i.replace("[", '')
            i = i.replace("]", '')
            if '1' in i:
                l.append(int(i))
        sequence.append(l)
        stimDur.append(row['stimDur'])
    key = np.array(key)
    sequence = np.array(sequence)
    stimDur = np.array(stimDur)
    allData = {'rt':rt, 'count': count, 'key':key, 'sequence': sequence, 'stimDur': stimDur}
    return allData



# get curry data and bandpass filter
eegFiles = os.listdir(eegdir + 's' + subj)
print(eegFiles)

fileName = subj + '_ses'+ str(ses)
fileName = [f for f in eegFiles if fileName in f and 'dpa' not in f and 'ceo' not in f]
fileName.sort()
print('reading' + str(fileName))


def extractData(filename):
    currydata, eegdata, sr, photocell, labels, chanloc = getEEG(eegdir + 's' + subj + '/' + filename)
    eegdata = filter(eegdata,sr,low = 45,low_stop = 55, high =1, high_stop = 0.25,gs=10)

    # pick a time winodw to make sure everything is recorded
    plt.plot(eegdata[int(10 * len(eegdata)/30): int(11 * len(eegdata)/30),80:110])
    plt.show()

    photocell = photocell[:,1]
    plt.plot(photocell[int(10 * len(eegdata)/30): int(11 * len(eegdata)/30)])
    plt.show()

    # truncate the data into blocks
    pcpos = np.where((np.diff(photocell) > 0.8e5)==True)[0]
    pcpos0 = np.append(pcpos[0:1], pcpos[1:][np.diff(pcpos) !=1])
    print('photocell', pcpos0.shape)

    events = currydata['events'].copy()

    val, count = np.unique(events[:,1],return_counts=True)
    stimonset = events[:,0][events[:,1] == 1.200002e+06]
    print('stimonset, ', stimonset.shape)
    # plt.scatter(pcpos0,stimonset) # to show that stimonset and stimtracker ones are the same
    # print(stimonset - pcpos0)
    # recode the events
    # 1200002 (with 16384) is stim onset
    # 1200001 (with 32768) is block onset
    # 1200001 (with 2048) is left key
    # 1 (no release) is right key
    # here we don't really care about block onset, because we don't know what key they press to continue anyway.

    myevent = np.zeros_like(events[:,0:2])
    myevent[events[:, 1] == 1.000000e+00,1] = 1
    myevent[events[:, 1] == 1.200002e+06,1] = 0
    myevent[events[:, 1] == 1.200001e+06,1] = -1
    myevent[:,0] = events[:,0]
    recordID = filename[-5]
    return myevent,  currydata, eegdata, sr, photocell, labels, chanloc, stimonset,pcpos0, recordID

if ses == 1:
    mydf = df1.copy()    # set the df2
if ses ==2:
    mydf = df2.copy()    # set the df2


myevent,  currydata, eegdata, sr, photocell, labels, chanloc, stimonset,pcpos, nRecord = extractData(fileName[0])

# uncomment this line for s106, block 0.
# realTstart = int(pcpos[7]-5*sr)
# stimonset = stimonset[stimonset>realTstart]
# stimonset = stimonset - realTstart
# pcpos = pcpos[pcpos>realTstart]
# pcpos -= realTstart
# photocell = photocell[realTstart:]
# eegdata = eegdata[realTstart:,:]
# myevent = myevent[myevent[:,0]>realTstart,:]
# myevent[:,0] = myevent[:,0] - realTstart

df_b = mydf.iloc[int(nRecord)*250: int(nRecord)*250+250]
trialPerBlock = 50
numBlock = int(len(stimonset)/trialPerBlock)
print('number of Block,', numBlock)
print('record ID,', int(nRecord))





# photocell = np.hstack((photocell0,photocell1))
#
# # tstart = len(photocell0)
# # myevent1[:,0] = myevent1[:,0] + tstart
#
# # stimonset1 += tstart
# myevent = np.vstack((myevent0, myevent1))
# stimonset = np.hstack((stimonset0, stimonset1))
# eegdata = np.vstack((eegdata0, eegdata1))






# combine the two sessions
condList = df_b['stimDur'].to_numpy()
df_b = getAllArrays(df_b.copy())   # no rt trials are -999. press 5 is 1, other wise is 0.

savedir = eegdir + subj
maxCount = 30
for i in range(0,numBlock):
    print('block' + str(i))
    blockCond = condList[i*50]
    print(blockCond)
    epochDur = sr*(blockCond * maxCount + 1 + 1) #1s before the chain (0.4-0.8s) fixation, before that was
                                                 # isi post response from alst trial, which is (1-1.4s) # , maxchain,
    # because subjects respond to different length

    # will fill the 1s postRT with nan for better ICA
    trials = np.zeros((trialPerBlock,int(epochDur),128))
    pctrials = np.zeros((trialPerBlock, int(epochDur)))
    rt = np.zeros((trialPerBlock))
    resp = np.zeros((trialPerBlock))
    for t in range(len(trials)):
        t0 = int(stimonset[i*50 + t] - 1*sr)
        t1 = int(t0+epochDur)
        trials[t,:] = eegdata[t0:t1,:]
        pctrials[t,:] = photocell[t0:t1]
        try:
            keysInd = np.where((myevent[:,0]>stimonset [i*50+t]) & (myevent[:,0]<stimonset[i*50+t+1]))
        except IndexError:
            keysInd = np.where((myevent[:,0]>stimonset [i*50+t]) & (myevent[:,0]<stimonset [i*50+t] + epochDur))

        key, pressTime = myevent[keysInd,1], (1000/sr)*(myevent[keysInd,0]-stimonset [i*50+t])
        key, pressTime = key[pressTime<=(blockCond*30+1) * sr], pressTime[pressTime<=(blockCond*30+1) * sr]
                        # if key is presssed within a second
        if key.size ==0:
            key = np.array(-999)
            pressTime = np.array(-999)
        if key.size>1:
            key = np.array(key[0])
            pressTime = np.array(pressTime[0])

        rt[t] = int(pressTime)
        resp[t] = int(key)

    # cross check

    df_rt = df_b['rt'][i * 50:i * 50 + 50]
    df_rt[df_rt!=-999] *=  1000
    df_rt = df_rt.astype('int')
    df_count = df_b['count'][i * 50:i * 50 + 50]
    df_key =df_b['key'][i * 50:i * 50 + 50] * 2 -1
    df_key[df_rt==-999] = -999
    df_sequence = df_b['sequence'][i * 50:i * 50 + 50,:]
    df_stimDur = df_b['stimDur'][i * 50:i * 50 + 50]

    df_block = dict()
    df_block = {'df_rt':df_rt, 'df_count':df_count, 'df_key':df_key, 'df_sequence':df_sequence, 'df_stimDur': df_stimDur}


    if sum(df_key==resp) ==50:
        print('responses matched')
    else:
        print('something went wrong, check!!!\n matched', sum(df_key==resp))
        print(df_rt[np.where(df_key!=resp)])  # check if all the not matching ones are -999
    mydict = dict()

    mydict = {'rt':rt,'resp':resp,'data':trials[:,::2,:],'stimDur':blockCond,'sr':1000,'prestimDur':1000*1,'df':df_block,\
              'labels':labels, 'chanloc':chanloc, 'photocell':pctrials[:,::2]}
    fname = 's' + subj +'_epoched_' + 'ses' + str(ses) + '_' +'block' + str(int(nRecord)*5+i) +'_'+str(int(blockCond * 1000)) + '.mat'
    print('saving ' +fname + '...')
    hdf5storage.savemat(eegdir + 's' + subj + '/' +fname, mydict, format='7.3',
                        store_python_metadata=True)
    print('done!\n')



####### second recording same session


myevent,  currydata, eegdata, sr, photocell, labels, chanloc, stimonset,pcpos, nRecord = extractData(fileName[1])

# # uncomment this line for s106, block 0.
# realTstart = int(pcpos0[7]-5*sr)
# stimonset0 = stimonset0[stimonset0>realTstart]
# pcpos0 = pcpos0[pcpos0>realTstart]
# photocell0 = photocell0[realTstart:]
# eegdata0 = eegdata0[realTstart:,:]
# myevent0 = myevent0[myevent0[:,0]>realTstart,:]
df_b = mydf.iloc[int(nRecord)*250: int(nRecord)*250+250]
trialPerBlock = 50
numBlock = int(len(stimonset)/trialPerBlock)
print('number of Block,', numBlock)
print('record ID,', int(nRecord))





# photocell = np.hstack((photocell0,photocell1))
#
# # tstart = len(photocell0)
# # myevent1[:,0] = myevent1[:,0] + tstart
#
# # stimonset1 += tstart
# myevent = np.vstack((myevent0, myevent1))
# stimonset = np.hstack((stimonset0, stimonset1))
# eegdata = np.vstack((eegdata0, eegdata1))






# combine the two sessions
condList = df_b['stimDur'].to_numpy()
df_b = getAllArrays(df_b.copy())   # no rt trials are -999. press 5 is 1, other wise is 0.

savedir = eegdir + subj
maxCount = 30
for i in range(0,numBlock):
    print('block' + str(i))
    blockCond = condList[i*50]
    print(blockCond)
    epochDur = sr*(blockCond * maxCount + 1 + 1) #1s before the chain (0.4-0.8s) fixation, before that was
                                                 # isi post response from alst trial, which is (1-1.4s) # , maxchain,
    # because subjects respond to different length

    # will fill the 1s postRT with nan for better ICA
    trials = np.zeros((trialPerBlock,int(epochDur),128))
    pctrials = np.zeros((trialPerBlock, int(epochDur)))
    rt = np.zeros((trialPerBlock))
    resp = np.zeros((trialPerBlock))
    for t in range(len(trials)):
        t0 = int(stimonset[i*50 + t] - 1*sr)
        t1 = int(t0+epochDur)
        trials[t,:] = eegdata[t0:t1,:]
        pctrials[t,:] = photocell[t0:t1]
        try:
            keysInd = np.where((myevent[:,0]>stimonset [i*50+t]) & (myevent[:,0]<stimonset[i*50+t+1]))
        except IndexError:
            keysInd = np.where((myevent[:,0]>stimonset [i*50+t]) & (myevent[:,0]<stimonset [i*50+t] + epochDur))

        key, pressTime = myevent[keysInd,1], (1000/sr)*(myevent[keysInd,0]-stimonset [i*50+t])
        key, pressTime = key[pressTime<=(blockCond*30+1) * sr], pressTime[pressTime<=(blockCond*30+1) * sr]
                        # if key is presssed within a second
        if key.size ==0:
            key = np.array(-999)
            pressTime = np.array(-999)
        if key.size>1:
            key = np.array(key[0])
            pressTime = np.array(pressTime[0])

        rt[t] = int(pressTime)
        resp[t] = int(key)

    # cross check

    df_rt = df_b['rt'][i * 50:i * 50 + 50]
    df_rt[df_rt!=-999] *=  1000
    df_rt = df_rt.astype('int')
    df_count = df_b['count'][i * 50:i * 50 + 50]
    df_key =df_b['key'][i * 50:i * 50 + 50] * 2 -1
    df_key[df_rt==-999] = -999
    df_sequence = df_b['sequence'][i * 50:i * 50 + 50,:]
    df_stimDur = df_b['stimDur'][i * 50:i * 50 + 50]

    df_block = dict()
    df_block = {'df_rt':df_rt, 'df_count':df_count, 'df_key':df_key, 'df_sequence':df_sequence, 'df_stimDur': df_stimDur}


    if sum(df_key==resp) ==50:
        print('responses matched')
    else:
        print('something went wrong, check!!!\n matched', sum(df_key==resp))
        print(df_rt[np.where(df_key!=resp)])  # check if all the not matching ones are -999
    mydict = dict()

    # resample because sr is 1250
    if sr !=2000:
        print('wrong sr!!!')
        import scipy
        trials = scipy.signal.resample(trials, int(epochDur * 1000/sr), axis= 1)
        pctrials = scipy.signal.resample(pctrials, int(epochDur * 1000/sr), axis= 1)

    mydict = {'rt':rt,'resp':resp,'data':trials[:,::2,:],'stimDur':blockCond,'sr':1000,'prestimDur':1000*1,'df':df_block,\
              'labels':labels, 'chanloc':chanloc, 'photocell':pctrials[:,::2]}
    fname = 's' + subj +'_epoched_' + 'ses' + str(ses) + '_' +'block' + str(int(nRecord)*5+i) +'_'+str(int(blockCond * 1000)) + '.mat'
    print('saving ' +fname + '...')
    hdf5storage.savemat(eegdir + 's' + subj + '/' +fname, mydict, format='7.3',
                        store_python_metadata=True)
    print('done!\n')
