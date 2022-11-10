# Created on 11/22/21 at 6:46 AM 

# Author: Jenny Sun
import os
import random

import numpy as np
import matplotlib.pyplot as plt
# the following import is required for matplotlib < 3.2:
from mpl_toolkits.mplot3d import Axes3D  # noqa
import mne
import numpy as np
from hdf5storage import loadmat
import scipy.io
from bipolar import hotcold
hotcoldmap = hotcold(neutral=0, interp='linear', lutsize=2048)   #black is the darkest
from preprocess.chanset import *





def plottopomap(data, ax, cmap=hotcoldmap,headmodel=False):
    '''inputs: '''
    data = np.reshape(data,(-1,1))
    labels, locdicchan, chans = chansets_neuroscan()
    mask = np.ones(128,dtype='bool')
    mask[chans] = False
    data[mask,0] = 0
    # labels = np.array(labels)[chans]
    nasion_chan = 1
    lpa_chan = 87
    rpa_chan = 90

    chan_pos = dict()

    for i, j in enumerate(locdicchan):
        chan_pos[labels[i]] = j
        # if i in chans.tolist():   #only pick 106
            # chan_pos[str(i)] = j * 0.1
            # chan_pos[labels[i]] = j
    mont = mne.channels.make_dig_montage(ch_pos=chan_pos,
                                         nasion=locdicchan[nasion_chan] , lpa=locdicchan[lpa_chan] ,
                                         rpa=locdicchan[rpa_chan] )

    basic_info = mne.create_info(ch_names=mont.ch_names, sfreq=500,
                                ch_types='eeg')
    evoked = mne.EvokedArray(data, basic_info)

    #
    evoked.set_montage(mont)
    # #
    if headmodel:
        mont.plot(show_names=True)

    fig, ax = plt.subplots()
    im, cm = mne.viz.plot_topomap(evoked.data[:,0], evoked.info, axes=ax,names=basic_info['ch_names'],cmap=cmap, show=False,\
                             extrapolate='local', contours=6)
    fig.show()
    return im,cm
