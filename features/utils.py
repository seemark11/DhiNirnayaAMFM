#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
from scipy.signal import find_peaks
import numpy as np


def prepare_meta(wav_path, dialect):
    fd1 = open(wav_path, 'r')
    lines = fd1.readlines()
    
    spk_all = []
    dia_all = []
    gen_all = []
    age_all = []
    filenamee = []
    
    for line in lines:
        line2read = line.strip('\n')
        print(line2read)
        
        utt = line2read.split('/')[-1]
            
        utt = utt.split('.')[0]
        spk = utt.split('_')[0]
        gen = utt.split('_')[1]
        age = utt.split('_')[2]
        dia = dialect
    
        #listn = [spk, dia, gen]
        spk_all.append(spk)
        dia_all.append(dia)
        gen_all.append(gen)
        age_all.append(age)
        filenamee.append(utt)
        
    return filenamee, spk_all, dia_all, gen_all, age_all

def compute_threshold_feature(mag_all, threshold):
    
    a_len = []
    a_mean = []
    a_var = []

    for ii in range(len(mag_all)):
        
        fre = mag_all[ii][0] 
        mag = mag_all[ii][1]
    
        peaks, _ = find_peaks(mag)
        freq, magg = fre[peaks], mag[peaks]
    
        magR_sci = magg[magg > threshold]
        freR_sci = freq[magg > threshold]
        
        freR_sci_mean = float("{0:.3f}".format(np.mean(freR_sci)))
        freR_sci_var = float("{0:.3f}".format(np.var(freR_sci)))

        a_len.append(len(magR_sci))
        a_mean.append(freR_sci_mean)
        a_var.append(freR_sci_var)
    
    return a_len, a_mean, a_var

def compute_slope_feature(mag_all):
    
    slope_all = []

    for ii in range(len(mag_all)):
        
        fre = mag_all[ii][0] 
        mag = mag_all[ii][1]
        
        slope, intercept = np.polyfit(fre,mag,1)
        slope = float("{0:.4f}".format(slope))
        slope_all.append(slope)
    
    return slope_all


def compute_dct_feature(mag_all):
    
    dct_values_all = []

    for ii in range(len(mag_all)):
        
        fre = mag_all[ii][0] 
        mag = mag_all[ii][1]
        
        #slope, intercept = np.polyfit(fre,mag,1)
        dct_values = scipy.fft.dct(mag, type=2)
        dct_values = dct_values[:10]
        #slope = float("{0:.4f}".format(slope))
        dct_values_all.append(dct_values)
    
    return dct_values_all

def compute_RF(mag_all, rhythmcount):
    
    RF_all = []

    for ii in range(len(mag_all)):
        
        freqsegment = mag_all[ii][0] 
        magsegment = mag_all[ii][1]
        
        peaks, _ = find_peaks(magsegment)
        freq, magg = freqsegment[peaks], magsegment[peaks]
    
        ranked = np.argsort(magg)
        top_magg_indx = ranked[::-1][:rhythmcount]
        #magg_top = magg[top_magg_indx]
        freq_top = freq[top_magg_indx]

        RF_all.append(freq_top)
    
    return RF_all
