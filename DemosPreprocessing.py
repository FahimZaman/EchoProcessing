#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:52:10 2024

@author: fazaman
"""

import os
from utilities import tools
cwd=os.getcwd()

#%%
# image class/category
CATEGORY = 'STEMI'
#image ID
IDX = 1

#%% Example Dense Flow
tools.demo_dense_optical_flow(idx=IDX, category=CATEGORY)

#%% Example Sparse Flow
mag, lon, rad = tools.demo_sparse_optical_flow(idx=IDX, category=CATEGORY)
    
#%% Example Cycle Extraction
tools.demo_cycle_extraction(idx=IDX, category=CATEGORY)

#%% ROI selection
tools.demo_select_ROI(idx=IDX, category=CATEGORY)