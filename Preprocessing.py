#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:18:30 2024

@author: fazaman
"""
import os
from utilities import tools

#%% Compress Videos
RAW_path = './Data/RAW'
Compressed_path = './Data/Compressed'

tools.video_compression(RAW_path, Compressed_path)

#%% Cleaning Artifacts
# Compressed_path = './Data/Compressed'
# Processed_path = './Data/Processed'

Compressed_path = './Data/SingleCycleCompressed'
Processed_path = './Data/SingleCycleProcessed'

tools.artifact_clean_resized(Compressed_path, Processed_path, DATASET='internal')

#%% Dense Optical Flow Calculation
# Processed_path = './Data/Processed'
# OpticalFlow_path = './Data/OpticalFlow'

Processed_path = './Data/SingleCycleInterpolated'
OpticalFlow_path = './Data/SingleCycleDenseOpticalFlow'

tools.calculate_dense_optical_flows(Processed_path, OpticalFlow_path, BOUND=15)

#%% Single Cycle Extraction From Compressed Manual
# Compressed_path = './Data/Compressed'
# SingleCycle_path = './Data/SingleCycleCompressedManual'

Compressed_path = './Data/Compressed'
SingleCycle_path = './Data/SingleCycleCompressed'

tools.single_cycle_extraction(Compressed_path, SingleCycle_path, automatic=False, st_frame=False, numpy=False, interpolation=False, color=True)

#%% Single Cycle Extraction From Compressed Manual
# Compressed_path = './Data/Compressed'
# SingleCycle_path = './Data/SingleCycleCompressedManual'

Compressed_path = './Data/SingleCycleCompressed'
SingleCycle_path = './Data/SingleCycleDiastolic'

tools.single_cycle_extraction(Compressed_path, SingleCycle_path, automatic=False, st_frame=True, numpy=True, interpolation=False, color=True)

#%% Single Cycle Extraction From Processed Automatic
Compressed_path = './Data/Processed'
SingleCycle_path = './Data/SingleCycleProcessedAutomatic'

tools.single_cycle_extraction(Compressed_path, SingleCycle_path)

#%% Diastolic Cycle Extraction From Processed Automatic
Compressed_path = './Data/Processed'
SingleCycle_path = './Data/DiastolicAutomatic'

tools.single_cycle_extraction(Compressed_path, SingleCycle_path, automatic=True, cycle='diastolic')

#%% Single Cycle Interpolation Automatic
# Processed_path = './Data/SingleCycleProcessedAutomatic'
# Interpolated_path = './Data/SingleCycleInterpolatedAutomatic'

Processed_path = './Data/SingleCycleProcessed'
Interpolated_path = './Data/SingleCycleInterpolated'

tools.single_cycle_interpolation(Processed_path, Interpolated_path, FRAME=32)

#%% Single Cycle Sparse Optical Flow

Interpolated_path = './Data/SingleCycleInterpolated'
Numpy_path = './Displacements'

tools.calculate_sparse_optical_flows(Interpolated_path, Numpy_path, ws=75, mlvl=5)

#%% ROI selection
Interpolated_path = './Data/SingleCycleInterpolatedAutomatic'
ROI_path = './Data/SingleCycleInterpolatedAutomaticROI'

tools.select_ROI(Interpolated_path, ROI_path, HEIGHT=128, WIDTH=160)

#%% Nifti Write
Processed_path = './Data/SingleCycleInterpolated'
Nifti_path = './Data/Nifti'

tools.write_NIFTI(Processed_path, Nifti_path)
