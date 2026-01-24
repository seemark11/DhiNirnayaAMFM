#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 23:19:05 2023

@author: self
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
# import pysptk
import scipy
import pyworld as pw
#%%
def f0estimate_rapt(path, fs):
    data_audio, fs = librosa.load(path, sr=fs, dtype="float64") 
    data_audio=data_audio-np.mean(data_audio)
    data_audio=data_audio/float(np.max(np.abs(data_audio)))

    size_frame=0.03 # sec
    size_step=0.01 # sec
    minf0=60 #hz
    maxf0=350 #hz
    size_frameS=size_frame*float(fs)
    size_stepS=size_step*float(fs)
    overlap=size_stepS/size_frameS
    #nF=int((len(data_audio)/size_frameS/overlap))-1
    data_audiof=np.asarray(data_audio, dtype=np.float64)

    # Calculate frame rate
    frame_rate = fs / (overlap * size_frameS)
    #print("Frame rate for F0:", frame_rate, "Hz")
    # F0=pysptk.sptk.rapt(data_audiof*(2**15), fs, int(size_stepS), min=minf0, max=maxf0, voice_bias=voice_bias, otype='f0')
    
    # Step size in seconds (convert samples -> seconds)
    hop_time = size_stepS / fs

    # Extract F0 using pyworld.harvest
    _f0, t = pw.harvest(data_audiof, fs, f0_floor=minf0, f0_ceil=maxf0)

    # Optional: refine using StoneMask (more accurate pitch tracking)
    f0_refined = pw.stonemask(data_audiof, _f0, t, fs)

    # Resample to match your frame step
    # Create new time axis with your hop size
    t_new = np.arange(0, len(data_audiof) / fs, hop_time)

    # Interpolate F0 to the new time grid
    F0 = np.interp(t_new, t, f0_refined, left=0, right=0)

    return F0, frame_rate, size_stepS
    
#%%