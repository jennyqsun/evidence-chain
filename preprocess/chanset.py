# Created on 11/8/22 at 8:50 PM 

# Author: Jenny Sun
import numpy as np
import pickle

def chansets_neuroscan():
    '''
    uses the pkl file to get the labels, positions and channels of neuroscan 128 channel cap
    :return: labels, positions, and channel index excluding the outer ring channels.
    example: labels, pos, chans = chansets_neuroscan()
    '''
    d = pickle.load(open('chanloc_neuroscan.pkl','rb'))
    chanlist = np.array((1,0,2,66,70,77,78,5,15,16,24,43,44,115,116,121,126,62,63,127,87,90))
    chans = np.arange(0, 128)
    chans = np.delete(chans, chanlist)
    return d['labels'], d['pos'],chans

if __name__ == '__main__':
    chansets_neuroscan()