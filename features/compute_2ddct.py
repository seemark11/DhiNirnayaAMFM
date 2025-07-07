#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 21:45:08 2022

@author: administrator
"""

from scipy.fftpack import dct
aaa = ammagarray[:,1:]
dd = dct(dct(aaa[0:,:].T).T)
dd = dd[0:3, 0:3]
dd = dd.flatten()