# Created on 11/9/22 at 2:29 PM 

# Author: Jenny Sun


# Created on 11/9/22 at 11:21 AM

# Author: Jenny Sun
# Created on 11/9/21 at 11:02 AM

# Author: Jenny Sun


import os
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import interactive
interactive(True)
from nn_model import *
import numpy as np
import os
import hdf5storage
from sklearn.model_selection import train_test_split
from pytorchtools import EarlyStopping, RMSLELoss
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

# local packages
# top level


# local packages subdir
from utils.topo import *
from utils.get_filt import *
from utils.sinc_fft import *
from utils.normalize import *
from utils.save_forward_hook import *
from utils.zscore_training import *




torch.cuda.device_count()
gpu0  = torch.device(0)
gpu1 = torch.device(1)
torch.cuda.set_device(gpu1)
device = torch.device(gpu1)
print(gpu0,gpu1)

import time
t1 = time.time()
############################# define random seeds ###########################

seednum = 2022
torch.manual_seed(seednum)
np.random.seed(seednum)
random.seed(seednum)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seednum)

############################ define model parameters ######################
timestart = 50   # 0 means starts from stimlus
timeend = 550
trialdur = timeend * 2 - timestart * 2


cond = 100   #100,250, or 500
notrainMode = False     # if true, just load the model
                        # if false, train model

saveForwardHook = True
if notrainMode:
    keepTrainMode = False
    createConfig = False
    saveForwardHook=False
else:
    createConfig = True    # when training, create config files.
    keepTrainMode = False  # set this to be True if wants to keep training from previous model
    zScoreData = False  # if tranining, default to save Forward Hook

datapath = '/ssd/rwchain-all/round2/rwchain-eeg/'
sr = 500
# timeend = 800 # when 300ms after stim

# Hyper-parameters
num_epochs = 300
batch_size = 64
learning_rate = 0.001
num_chan = 98
dropout_rate = 0.7
compute_likelihood = True
EarlyStopPatience = 8
weight_decay = 0

######################## tensorbaord initilization ###########################

model_0 = peakValue(dropout=dropout_rate).to(device)
model = peakValue(dropout=dropout_rate).to(device)

model_0 = torch.nn.DataParallel(model_0, device_ids = [1])
model = torch.nn.DataParallel(model, device_ids = [1])

######################## creating directory and file nmae ############for s########


postname = '_peak_value'


modelpath = 'model_out/trained_model' + postname
resultpath = 'model_out/results' + postname
figurepath = 'model_out/figures' + postname

isExist = os.path.exists(modelpath)
isExist = os.path.exists(modelpath)
if not isExist:
    os.makedirs(modelpath)
    print(modelpath + ' created')

isExist = os.path.exists(figurepath)
if not isExist:
    os.makedirs(figurepath)
    print(figurepath + ' created')

isExist = os.path.exists(resultpath)
if not isExist:
    os.makedirs(resultpath)
    print(resultpath + ' created')


####################### some functions for getting the EEG data ##############
def viz_histograms(model, epoch):
    for name, weight in model.named_parameters():
        try:
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)
        except NotImplementedError:
            continue


def getIDs(path):
    allDataFiles = os.listdir(path)
    finalsub = [i for i in allDataFiles if "s104" not in i and "txt" not in i]
    finalsub.sort()
    return np.unique(finalsub)



def getAllCondNames(datapath, subIDs, cond):
    '''this function reutrns a list absoluate path to the files for a certain condition'''
    fnames = []
    ids = []
    for j,s in enumerate(subIDs):
        subFiles = os.listdir(datapath + s)
        subFiles = [f for f in subFiles if '_clean_' in f and '_'+str(cond) in f]
        ids.extend([j]*len(subFiles))
        subFiles = [datapath + s + '/' + f for f in subFiles]

        fnames.extend(subFiles)
        print(len(subFiles), 'files found for', s, 'for condition', cond)
    print(len(fnames), 'files found in total')
    return fnames, ids

# def loadsubjdict(path, fnames):
#     datadict = hdf5storage.loadmat(path + subID+ '_high' + '.mat')
#     return datadict
def loadsubjdict(fnames):
    datadict = hdf5storage.loadmat(fnames)
    return datadict

def reshapedata(data):
    timestep, nchan, ntrial = data.shape
    newdata = np.zeros((ntrial, nchan, timestep))
    for i in range(0, ntrial):
        newdata[i, :, :] = data[:, :, i].T
    return newdata

class SubTrDataset(Dataset):
    def __init__(self, transform=None):
        self.n_samples = X_train_sub.shape[0]
        self.x_data = np.asarray(X_train_sub, dtype=np.float32)
        Xmean = np.mean(self.x_data, axis=2)
        Xmean_mat = Xmean[:, :, np.newaxis].repeat(X_train_sub.shape[-1], axis=2)
        self.x_data = self.x_data - Xmean_mat

        self.y_data = np.asarray(y_train_sub, dtype=np.float32)

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[[index]]

        if self.transform:  # if transform is not none
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


# produce the dataset
class ValDataset(Dataset):
    def __init__(self, transform=None):
        self.n_samples = X_val.shape[0]
        self.x_data = np.asarray(X_val, dtype=np.float32)
        Xmean = np.mean(self.x_data, axis=2)
        Xmean_mat = Xmean[:, :, np.newaxis].repeat(X_val.shape[-1], axis=2)
        self.x_data = self.x_data - Xmean_mat
        self.y_data = np.asarray(y_val, dtype=np.float32)

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[[index]]

        if self.transform:  # if transform is not none
            sample = self.transform(sample)
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


# produce the dataset
class TrDataset(Dataset):
    def __init__(self, transform=None):
        self.n_samples = X_train0.shape[0]
        self.x_data = np.asarray(X_train0, dtype=np.float32)
        Xmean = np.mean(self.x_data, axis=2)
        Xmean_mat = Xmean[:, :, np.newaxis].repeat(X_train0.shape[-1], axis=2)
        self.x_data = self.x_data - Xmean_mat

        self.y_data = np.asarray(y_train0, dtype=np.float32)

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[[index]]

        if self.transform:  # if transform is not none
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


# produce the dataset
class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.n_samples = X_test.shape[0]
        self.x_data = np.asarray(X_test, dtype=np.float32)
        Xmean = np.mean(self.x_data, axis=2)
        Xmean_mat = Xmean[:, :, np.newaxis].repeat(X_test.shape[-1], axis=2)
        self.x_data = self.x_data - Xmean_mat
        self.y_data = np.asarray(y_test, dtype=np.float32)

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[[index]]

        if self.transform:  # if transform is not none
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):  # not it became a callable object
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
        print('init xavier uniform %s' % m)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        print('init xavier uniform %s' % m)
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)




class weightConstraint(object):
    def __init__(self, model):
        self.cutoff = model.module.cutoff
        self.min_freq = 1
        self.min_band = 2
        self.b1max = int(self.cutoff - self.min_freq -  self.min_band)
    def __call__(self, module):
        if hasattr(module, 'filt_b1'):
            b1 = module.filt_b1.data
            band = module.filt_band.data
            fs = module.freq_scale
            b1 = b1.clamp(-1 * (torch.min(torch.abs(b1)+(self.min_freq /fs)+torch.abs(band),
                                                            torch.ones_like(band) * self.b1max/fs)), torch.min(torch.abs(b1)+(self.min_freq /fs)+torch.abs(band),
                                                            torch.ones_like(band) * self.b1max/fs))
            module.filt_b1.data = b1


################################### CREATE CONFIG FILES ###################
if createConfig:
    config_object = ConfigParser()
    config_object["data"] = {
        "dataset": datapath,
        "filters": [1,45],
        "sr": sr,
        "zscore": zScoreData
    }

    config_object["hyperparameters"] = {
        "N_filters":model.module.num_filters,
        "filter_length": model.module.filter_length,
        "pool_window_ms": model.module.pool_window_ms,
        "stride_window_ms": model.module.stride_window_ms,
        "attentionLatent": model.module.attentionLatent,
        "N_chan":model.module.num_chan,
        "patience":EarlyStopPatience
    }

    config_object["optimization"] = {
        "batch_size": batch_size,
        "maxepoch": num_epochs,
        "seed": seednum,
        "weights_constrain": model.module.cutoff
    }
    config_object["Notes"] = {
        "notes": 'this version is created when weights are clamped at forward pass of both f1 and f2',
    }
    #Write the above sections to config.ini file
    with open(modelpath + '/config.ini', 'w') as conf:
        config_object.write(conf)
else:
    #Write the above sections to config.ini file
    config_object = ConfigParser()
    config_object.read(modelpath + "/config.ini")
    zScoreData = config_object["data"]["zscore"] == 'True'
    datapath = config_object["data"]["dataset"]
    EarlyStopPatience = int(config_object["hyperparameters"]["patience"])
    seednum = int(config_object["optimization"]["seed"])


def getDictData(subdict):
    goodtrials = subdict['goodtrials']
    data = subdict['data']
    goodchans = subdict['goodchans']
    maskchan = subdict['maskchan']

    resp = subdict['resp'][goodtrials]
    rt = subdict['rt'][goodtrials]
    stimDur = subdict['stimDur']
    verybadchans = subdict['verybadchans']
    sequence = subdict['df']['df_sequence'][goodtrials,:]
    return data, rt, resp, sequence,maskchan,verybadchans

# %%
############################################################################
################################# starts here ###############################
############################################################################
finalsubIDs = getIDs(datapath)
tbdir = 'runs/' + postname
if notrainMode is False:
    try:
        len(os.listdir(tbdir)) != 0
        shutil.rmtree(tbdir)
    except:
        pass
    tb = SummaryWriter(tbdir)

fnames, subid = getAllCondNames(datapath, finalsubIDs, cond)  # subid is unique value of each subject
subdict = loadsubjdict(fnames[0])
sr = subdict['sr']
maxdur = cond*30+1000+1000
dataall = np.zeros((0,maxdur,128))
rtall = np.zeros(0)
respall = np.zeros(0)
sequenceall = np.zeros((0,30))
maskchanall = np.zeros((0,128))
sidall = np.zeros_like(rtall)

for i, j in enumerate(fnames):
    print(i)
    subdict = loadsubjdict(j)
    data, rt, resp, sequence,maskchan,verybadchans = getDictData(subdict)
    dataall = np.vstack((dataall, data))
    rtall = np.hstack((rtall,rt))
    sidall = np.hstack((sidall, np.zeros_like(rt)+subid[i]))
    respall = np.hstack((respall, resp))
    sequenceall = np.vstack((sequenceall, sequence))
    maskchanall = np.vstack((maskchanall, maskchan))
# load all subjects dictionary




# let's check if there are enough ones that are consistently good
chansum = np.sum(maskchanall, axis=0)  # is block x 128
nblock = maskchanall.shape[0]
allgoodchans = np.where(chansum>=int(nblock*0.9))[0]
# if the channels are considered to be good 90% of the blocks
print(allgoodchans.shape[0], 'good chans consistently')

# crosscheck these with the outer rins
labels, locdicchan, goodchan = chansets_neuroscan()
finalgoodchans = np.intersect1d(allgoodchans, goodchan)

# down sample
data=dataall[:,::2,finalgoodchans]

# chop the data such that we have
# identify the frame where peak evidenc occur
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



# dictall = dict()
# dictall = {'data':data, 'rt':rtall, 'resp':respall, 'sequence': sequenceall, 'maskchan': maskchanall,\
#            'finalgoodchans':finalgoodchans,'condition':np.array(cond),'sr':sr}
# hdf5storage.savemat(datapath + 'combined/all_' + str(cond), dictall, format='7.3', \
#                     store_python_metadata=True)

    # let's train a model to look at 5 samples back if we can