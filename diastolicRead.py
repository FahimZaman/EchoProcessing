#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:49:55 2024

@author: fazaman
"""

import os
import matplotlib.pyplot as plt
from utilities import tools
import numpy as np
            
#%%
data_path = './Numpys'

tools.density_plot_relax(data_path, thr=0.27, plot=True, pvalue=True, errors=False, lineplot=False)

#%%
numpy_path = './Displacements'
diastolic_path = './Numpys'

tools.sparseFlow_displacement_interpolated(numpy_path, diastolic_path, plots=True, longplot=True, smoothing=[7,2])

#%%
filepathTTS = './accumulated_gradTTS.mp4'
filepathSTEMI = './accumulated_gradSTEMI.mp4'
gTTS = tools.read_video(filepathTTS, color=True)
w = int(gTTS.shape[2]/2)
gTTSs, gTTSf = gTTS[:,:,:w,:], gTTS[:,:,w:,:]
gMI = tools.read_video(filepathSTEMI, color=True)
gMIs, gMIf = gMI[:,:,:w,:], gMI[:,:,w:,:]

filepathTTS = './accumulated_imgGrad_TTS.mp4'
filepathSTEMI = './accumulated_imgGrad_STEMI.mp4'
iTTS = tools.read_video(filepathTTS, color=True)
iMI = tools.read_video(filepathSTEMI, color=True)

nplot = 9
t = np.linspace(0,31,nplot).astype(np.uint8)
gTTSshat = [gTTSs[i] for i in t]
gMIshat = [gMIs[i] for i in t]
gTTSfhat = [gTTSf[i] for i in t]
gMIfhat = [gMIf[i] for i in t]
gcats = [gTTSshat, gTTSfhat, gMIshat, gMIfhat]

# r, c = 1, nplot
# for ct in gcats:
#     plt.figure(figsize=(5*nplot,4))
#     for n in range(nplot):
#         plt.subplot(r,c,n+1)
#         plt.imshow(ct[n])
#         plt.axis('off')
#     plt.tight_layout()
    
# cm = plt.get_cmap('jet_r')
# csf = []
# for i in range(len(gcats)):
#     csf.append(cm(spatial_flow[i]))
# csf = np.array(csf, dtype=np.float32)[...,0:3]

save_dir = '/user/iibi/fazaman/Downloads/dataprocess/Milan_sonka/Figures/Takotsubo/Paper4/GradPlots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
nameCats = ['TTS_spatial', 'TTS_flow', 'STEMI_spatial', 'STEMI_flow']
for ct in range(len(gcats)):
    for n in range(nplot):
        fig= plt.figure(figsize=((gcats[ct][n].shape[1])/30,(gcats[ct][n].shape[0]/30)))
        plt.imshow(gcats[ct][n])
        plt.axis('off')
        plt.tight_layout()
        filename = os.path.join(save_dir,nameCats[ct]+f'_frame-{n:02d}.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    
# iTTSs, iTTSf = iTTS[:,:,:w,:], iTTS[:,:,w:,:]
# iMI = tools.read_video(filepathSTEMI, color=True)
# iMIs, iMIf = iMI[:,:,:w,:], iMI[:,:,w:,:]
# iTTSshat = [iTTSs[i] for i in t]
# iMIshat = [iMIs[i] for i in t]
# iTTSfhat = [iTTSf[i] for i in t]
# iMIfhat = [iMIf[i] for i in t]

# icats = [iTTSshat, iTTSfhat, iMIshat, iMIfhat]
# for ct in icats:
#     plt.figure(figsize=(5*nplot,4))
#     for n in range(nplot):
#         plt.subplot(r,c,n+1)
#         plt.imshow(ct[n])
#         plt.axis('off')
#     plt.tight_layout()

#%%
filepathTTSflow = './accumulated_flowTTS.mp4'
filepathSTEMIflow = './accumulated_flowSTEMI.mp4'

tools.motion_density_plot(filepathTTSflow, filepathSTEMIflow, nplot=9, smoothing=[5,2])
