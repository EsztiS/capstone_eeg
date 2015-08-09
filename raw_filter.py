# -*- coding: utf-8 -*-
"""
Created on Sun Aug 9 2015

based on the Beat the benchmark with CSP and Logisitic regression from
@author: alexandrebarachant

** Removed parallelization, as couldn't get to work.

General Idea :

The goal of this challenge is to detect events related to hand movements. Hand 
movements are caracterized by change in signal power in the mu (~10Hz) and beta
(~20Hz) frequency band over the sensorimotor cortex. 

Preprocessing :

Signal are bandpass-filtered between 7 and 30 Hz to catch most of the signal of
interest. 

"""

print(__doc__)

import numpy as np
import pandas as pd
from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP

from sklearn.linear_model import LogisticRegression
from glob import glob

from scipy.signal import butter, lfilter, convolve, boxcar

def creat_mne_raw_object(fname,read_events=True):
    """Create a mne raw instance from csv file"""
    # Read EEG file
    data = pd.read_csv(fname)
    
    # get chanel names
    ch_names = list(data.columns[1:])
    
    # read EEG standard montage from mne
    montage = read_montage('standard_1005',ch_names)

    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[ch_names]).T # From microvolts to volts
    
    if read_events:
        # events file
        ev_fname = fname.replace('_data','_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T
        
        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data,events_data))
        
    # create and populate MNE info structure
    info = create_info(ch_names,sfreq=500.0, ch_types=ch_type, montage=montage)
    info['filename'] = fname
    
    # create raw object 
    raw = RawArray(data,info,verbose=False)
    
    return raw

subjects = range(1,2)
ids_tot = []

# design a butterworth bandpass filter 
freqs = [7, 30]
b,a = butter(5,np.array(freqs)/250.0,btype='bandpass')

# convolution
# window for smoothing features
nwin = 250

# training subsample
subsample = 10

cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

for subject in subjects:

    ################ READ DATA ################################################
    fnames =  glob('/Users/eszti/Documents/__NYC_DSA/Dev/capstone/train/subj%d_series*_data.csv' % (subject))
    
    # read and concatenate all the files
    raw = concatenate_raws([creat_mne_raw_object(fname) for fname in fnames])
       
    # pick eeg signal
    picks = pick_types(raw.info,eeg=True)
    
    # Filter data for alpha frequency and beta band
    # Note that MNE implement a zero phase (filtfilt) filtering not compatible
    # with the rule of future data.
    # Here we use left filter compatible with this constraint. 
    raw._data[picks] = lfilter(b,a,raw._data[picks])
    raw.to_data_frame()
    
    