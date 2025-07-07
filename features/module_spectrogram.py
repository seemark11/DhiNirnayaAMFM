
# System and library module import
import numpy as np
from scipy.signal import find_peaks
import sys, re
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
from datetime import datetime
import librosa
from scipy.signal import medfilt, hilbert
# RFA custom module import
from module_F0_v1 import *	# FM demodulation (F0 estimation, 'pitch' tracking)
#===============================================================
# Assign configuration parameters to variables (see .conf file)
from rfa_single_conf import *

def plotspectrogramheatmap(pltobj, freqarray, magarray, signalsecs, specfreqmin, specfreqmax, specgramdotsize, specheatmaptype, fontsize):

    # y-axis as scale of the range, number of spectra in spectrogram
    y = np.linspace(specfreqmin,specfreqmax,len(magarray[0]))

    # x-axis as signal time range, number of spectra in spectrogram
    x = np.linspace(0,signalsecs,len(magarray))

    # Colormap is derived from magnitudes at each frequency/spectrum
    for i, freqvals, magvals in zip(x, freqarray, magarray):
        freqvals = freqvals[1:]
        magvals = magvals[1:]
        x = [i] * len(freqvals)
        pltobj.scatter(
            x,freqvals, c=magvals, cmap=specheatmaptype,
            marker="s", s=specgramdotsize)

    # Spectrogram properties
    pltobj.set_xlim(0,np.ceil(signalsecs))
    sfmin = np.floor(specfreqmin)
    sfmax = np.ceil(specfreqmax)
    pltobj.set_ylim(sfmin,sfmax)
    pltobj.grid(b=True, which="major", axis="both")
    pltobj.set_xlabel("Time (s)", fontsize=fontsize)
    pltobj.set_ylabel("Freq (Hz)", fontsize=fontsize)

    return

def get_AM_spectrogram_all(file_path, specwindowsecs, specstrides):

    fd1 = open(file_path, 'r')
    lines = fd1.readlines()
    
    for line in lines:
        line2read = line.strip('\n')
        #print(line2read)
        fs, signal = wave.read(line2read)    # read sampling frequency and signal
        if len(signal.shape) == 2:
            signal = signal[:,1]
      
        signallength = len(signal)        # define numerical signal length
        signalseconds = signallength / fs    # define signal length in seconds
        signal = signal / max(abs(signal))    # scale signal: -1 ... 0 ... 1
        envelope = np.abs(hilbert(signal))
    	envelope = medfilt(envelope, 501)
    	envelope = envelope / np.max(envelope

        #===============================================================
        # Create AM spectrogram and max magnitude value trajectory
        ammagarray, amfreqarray, ammaxmags, ammaxfreqs = spectrogramarray(envelope, fs,amspecfreqmin, amspecfreqmax, specdownsample, 
                                                                          spectrumpower, specwindowsecs, specstrides)
        
        
        nan_mask = np.isnan(ammagarray)
        total_nans = np.sum(nan_mask)
        
        print(ammagarray.shape)
        print(total_nans)
        
    return


def get_FM_spectrogram_all(file_path, specwindowsecs, specstrides):

    fd1 = open(file_path, 'r')
    lines = fd1.readlines()
        
    for line in lines:
        line2read = line.strip('\n')
        line2read = line2read.split(',')[0]
        '''
        print(line2read)
        
        #fs, signal = wave.read(line2read)    # read sampling frequency and signal
        signal, fs = librosa.load(line2read, sr=16000)
        if len(signal.shape) == 2:
            signal = signal[:,1]
      
        signallength = len(signal)        # define numerical signal length
        signalseconds = signallength / fs    # define signal length in seconds
        signal = signal / max(abs(signal))    # scale signal: -1 ... 0 ... 1

        f0array, framerate, frameduration = f0estimate(signal, fs)
        '''
        f0array, framerate, frameduration = f0estimate_rapt(line2read, 16000)
        f0array = medfilt(f0array, 101)

        #===============================================================
        # Create FM spectrogram and max value trajectory
        #f0array = f0array[f0array != 0]
        fmmagarray, fmfreqarray, fmmaxmags, fmmaxfreqs = spectrogramarray(f0array, 
                                                                          framerate, 
                                                                          fmspecfreqmin, 
                                                                          fmspecfreqmax,
                                                                          specdownsample, 
                                                                          spectrumpower, 
                                                                          specwindowsecs, 
                                                                          specstrides)
        nan_mask = np.isnan(fmmagarray)
        total_nans = np.sum(nan_mask)
        
        print(fmmagarray.shape)
        print(total_nans)
    
    return 


def get_AM_spectrogram(file_path, specwindowsecs, specstrides):
    fs, signal = wave.read(file_path)    # read sampling frequency and signal
    if len(signal.shape) == 2:
        signal = signal[:,1]
      
    signal = signal / max(abs(signal))    # scale signal: -1 ... 0 ... 1
    signallength = len(signal)        # define numerical signal length
    signalseconds = signallength / fs    # define signal length in seconds
    envelope = np.abs(hilbert(signal))
    envelope = medfilt(envelope, 501)
    envelope = envelope / np.max(envelope)

    #===============================================================
    # Create AM spectrogram and max magnitude value trajectory
    ammagarray, amfreqarray, ammaxmags, ammaxfreqs = spectrogramarray_v1(envelope, fs,amspecfreqmin, amspecfreqmax, specdownsample, 
                                                                      spectrumpower, specwindowsecs, specstrides)
        
    return ammagarray


def get_FM_spectrogram(file_path, specwindowsecs, specstrides):

    '''
    signal, fs = librosa.load(file_path, sr=16000)
    if len(signal.shape) == 2:
        signal = signal[:,1]

    signal = signal / max(abs(signal))    # scale signal: -1 ... 0 ... 1
    signallength = len(signal)        # define numerical signal length
    signalseconds = signallength / fs    # define signal length in seconds

    f0array, framerate, frameduration = f0estimate(signal, fs)
    '''
    f0array, framerate, frameduration = f0estimate_rapt(file_path, 16000)
    f0array = medfilt(f0array, 101)

    #===============================================================
    # Create FM spectrogram and max value trajectory
    fmmagarray, fmfreqarray, fmmaxmags, fmmaxfreqs = spectrogramarray_v1(f0array, 
                                                                          framerate, 
                                                                          fmspecfreqmin, 
                                                                          fmspecfreqmax,
                                                                          specdownsample, 
                                                                          spectrumpower, 
                                                                          specwindowsecs, 
                                                                          specstrides)
    return fmmagarray
