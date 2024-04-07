#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:47:37 2024

@author: fazaman
"""

import os
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from skimage.measure import label
from scipy.ndimage import zoom
import imageio
import cv2
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from scipy import interpolate
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix, accuracy_score

#%% Video compression
def compress_video(video_full_path, output_file_name, target_bitrate=int(10*1e3)):
    import ffmpeg
    '''This function compresses the video to target bitrate using ffmpeg codec'''
    i = ffmpeg.input(video_full_path)
    ffmpeg.output(i, output_file_name, **{'c:v': 'libx264', 'b:v': target_bitrate, 'f': 'mp4'}).run()
    
def video_compression(input_path, output_path, target_bitrate=int(10*1e3)):
    '''This function compresses the video to target bitrate given video directory'''
    print('\n\n-----------------------------------')
    print('Video Compression')
    print('-----------------------------------\n')
    # check for output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # list all categories
    category_names = natsorted(os.listdir(input_path))
    category_list = [os.path.join(input_path,i) for i in category_names]
    for n in range(len(category_list)):
        input_filenames = [i for i in natsorted(os.listdir(category_list[n]))]
        output_directory = os.path.join(output_path, natsorted(os.listdir(input_path))[n])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Check for missing files
        if len(os.listdir(output_directory)) == len(input_filenames):
            print('All '+category_names[n]+' files are already compressed...')
        else:
            tag = '_compressed.mp4'
            output_filenames = natsorted([os.path.join(output_directory, os.path.splitext(i)[0]+tag) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            input_filenames = natsorted([os.path.join(category_list[n], i) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            print(category_names[n], f'[{len(output_filenames)} files to be processed]')
            print('-----------------------------------')
            print(input_filenames)
            print('-----------------------------------')
            # Compress videos
            for files in tqdm(range(len(input_filenames))):
                compress_video(input_filenames[files], output_filenames[files], target_bitrate)
                
#%% Artifact cleaning
def read_video(filepath, remove_color=False, color=False, colorCAT='RGB'):
    '''This function reads the video and removes the color artifacts'''
    cap = cv2.VideoCapture(filepath)
    video = []
    while(cap.isOpened()):
        ret, cvideo = cap.read()
        if ret == True:
            if remove_color==True:
                cvideo = color_remove(cvideo)
            if color==True:
                if colorCAT=='RGB':
                    cvideo = cv2.cvtColor(cvideo, cv2.COLOR_BGR2RGB)
                video.append(cvideo)
            else:
                video.append(cv2.cvtColor(cvideo, cv2.COLOR_RGB2GRAY))
        else:
            break
    cap.release()
    if color==False:
        video = standardize(np.array(video, dtype=np.float32))
    else:
        video = np.array(video, dtype=np.uint8)
    return video

# def color_remove(video):
#     '''This function removes the colored objects from the video'''
#     video = video.astype(np.uint16)
#     h = int(video.shape[0]*7/8)
#     B, G, R = video[...,0].copy(), video[...,1].copy(), video[...,2].copy()
#     B[:h], G[:h], R[:h] = 0, 0, 0
#     G1 = np.where(G>((B+R)/2),0,video[...,1])
#     R1 = np.where(G>((B+R)/2),0,video[...,2])
#     B1 = np.where(G>((B+R)/2),0,video[...,0])
#     video = np.stack((B1,G1,R1), axis=-1).astype(np.uint8)
#     return video

def color_remove(video):
    '''This function removes the colored objects from the video'''
    std = np.std(video, axis=-1)
    std = standardize(std)
    diff = np.stack([std, std, std], axis=-1)
    video = np.where(diff>0.1, 0, video)
    return video  

def standardize(video):
    '''This function does min-max standardization'''
    video = video - np.min(video)
    video = video / np.max(video)
    return video

def video_clean(video, thr=0.005):
    '''This function removes the grayscale artifacts given an outlier area threshold'''
    stdVol = np.std(standardize(video), axis=0)
    stdVol = outlier_remove(stdVol, thr)
    arrays = [stdVol for _ in range(len(video))]
    mask = np.stack(arrays, axis=0)
    video = np.multiply(video, mask)
    return video, stdVol

def outlier_remove(video, thr=0.005):
    '''This function removes the disconnected objects given an area threshold'''
    video = np.where(video>thr,1,0)
    lbl = label(video, connectivity=2)
    l, c = np.unique(lbl, return_counts=True)
    l, c = l[1:], c[1:]
    tlbl = np.argsort(c)[::-1][0]
    video = np.where(lbl==l[tlbl],video,0)
    return video

def mask_id(mask, height, width):
    '''This function sets the ROI in the middle of the video'''
    h, w = mask.shape
    hs, ws = np.where(mask==1)
    hs, ws = np.sort(hs), np.sort(ws)
    h1, h2, w1, w2 = hs[0], hs[-1], ws[0], ws[-1]
    st_h, en_h, st_w, en_w = 0, height, 0, width
    h1, w1 = h1-int((height-(h2-h1))/2), w1-int((width-(w2-w1))/2)
    h2, w2 = h1+height, w1+width
    if h1<0:
        h1=0
    if h2>h:
        h2=h
    if w1<0:
        w1=0
    if w2>w:
        w2=w
    en_h, en_w = h2-h1, w2-w1
    return h1, h2, w1, w2, st_h, st_w, en_h, en_w

def write_video(video, size, fps, filepath, color=False):
    '''This function writes video to output directory'''
    # video standardize
    video = (standardize(video)*255).astype(np.uint8)
    out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for fr in range(len(video)):
        if color==False:
            cvideo = np.stack([video[fr], video[fr], video[fr]], axis=-1)
        else:
            cvideo = video[fr]
        out.write(cvideo)
    out.release()
    
def write_nifti(video, filepath):
    '''This functtion writes nifti from video to output directory'''
    # video standardize
    video = standardize(video).astype(np.float32)
    output = sitk.GetImageFromArray(video)
    sitk.WriteImage(output, filepath)
    
def clean_video(input_file, output_file, HEIGHT=256, WIDTH=352, FPS=30, REDUCE=0.25, DATASET='internal'):
    '''This function removes the color objects & motionless pixels, resize the video and write to output directory'''
    video = read_video(input_file, remove_color=True)
    video, mask = video_clean(video, thr=0.005)
    if DATASET=='external':
        _, H, _ = video.shape
        video = zoom(video, (1, HEIGHT/H, HEIGHT/H))
        video = np.where(video<0,0,video)
        video = video - np.min(video)
        W = video.shape[2]
        if W<WIDTH:
            st = int((WIDTH-W)/2)
            svideo = np.zeros((len(video), HEIGHT, WIDTH))
            svideo[:, :, st:st+W] = video[:]
        else:
            st = int((W-WIDTH)/2)
            svideo = video[:, :, st:st+WIDTH]
    else:
        video = zoom(video, (1, REDUCE, REDUCE))
        video = np.where(video<0,0,video)
        mask = zoom(mask, (REDUCE, REDUCE))
        mask = np.where(mask>0,1,0)
        h1, h2, w1, w2, st_h, st_w, en_h, en_w = mask_id(mask, HEIGHT, WIDTH)
        svideo = np.zeros((len(video), HEIGHT, WIDTH))
        svideo[:, st_h:en_h,st_w:en_w] = video[:, h1:h2, w1:w2]
    svideo = standardize(svideo)
    svideo = (svideo*255).astype(np.uint8)
    write_video(svideo, (WIDTH, HEIGHT), FPS, output_file)

def artifact_clean_resized(input_path, output_path, HEIGHT=256, WIDTH=352, FPS=30, REDUCE=0.25, DATASET='internal'):
    '''This function remove all the artifacts from the video given the video directory'''
    print('\n\n-----------------------------------')
    print('Artifacts cleaning')
    print('-----------------------------------\n')
    # check for output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # list all categories
    category_names = natsorted(os.listdir(input_path))
    category_list = [os.path.join(input_path,i) for i in category_names]
    for n in range(len(category_list)):
        input_filenames = [i for i in natsorted(os.listdir(category_list[n]))]
        output_directory = os.path.join(output_path, natsorted(os.listdir(input_path))[n])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Check for missing files
        if len(os.listdir(output_directory)) == len(input_filenames):
            print('All '+category_names[n]+' files are already cleaned...')
        else:
            tag = '_cleanedResized.mp4'
            output_filenames = natsorted([os.path.join(output_directory, os.path.splitext(i)[0]+tag) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            input_filenames = natsorted([os.path.join(category_list[n], i) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            print(category_names[n], f'[{len(output_filenames)} files to be processed]')
            print('-----------------------------------')
            print(input_filenames)
            print('-----------------------------------')
            # Compress videos
            for files in tqdm(range(len(input_filenames))):
                clean_video(input_filenames[files], output_filenames[files], HEIGHT, WIDTH, FPS, REDUCE, DATASET)
                
#%% Optical flow calculation
def plot2video(saveimage, savepath, fps=30):
    '''This function writes a video from plot images'''
    # load plot images
    images = []
    for filename in sorted([i for i in os.listdir(saveimage) if i.endswith('png')]):
        images.append(imageio.imread(os.path.join(saveimage,filename)))
    images = np.array(images, dtype=np.uint8)
    R, G, B = images[...,0], images[...,1], images[...,2]
    images = np.stack((B, G, R), axis=-1)
    # write video
    write_video(images, (images.shape[2], images.shape[1]), fps, savepath, color=True)
    
def local_minima_optical_flow(video, filename=[''], cycle='full'):
    # optical flow calculation
    xflow, yflow, mag, angle = dense_flow(video, bound=15, filename=filename)
    ang0 = standardize(np.mean(angle, axis=(1,2)))
    # angle interpolation
    if len(ang0)>62:
        window_length = 31
    else:
        window_length = int(len(ang0)*0.5)
        if np.mod(window_length,2)==0:
            window_length+=1
    ang = savgol_filter(ang0, window_length, 2)
    localMaximas = list(argrelextrema(ang, np.greater, order=3)[0])
    localMaximasValues = [ang[i] for i in localMaximas]
    localMinimas = list(argrelextrema(ang, np.less, order=3)[0])
    localMinimasValues = [ang[i] for i in localMinimas]
    # end point calculation
    combined = localMaximas+localMinimas
    maxlocal = np.max(combined)
    if maxlocal in localMinimas:
        localMaximas.append(len(ang)-1)
        localMaximasValues.append(ang[-1])
    else:
        localMinimas.append(len(ang)-1)
        localMinimasValues.append(ang[-1])
    # start and frame
    if cycle=='full':
        startFrame, endFrame = localMinimas[0], localMinimas[1]
    elif cycle=='diastolic':
        startFrame = localMaximas[0]
        if localMaximas[0] > localMinimas[0]:
            endFrame = localMinimas[1]
        else:
            endFrame = localMinimas[0]
    return startFrame, endFrame, len(video)
    
    
def demo_dense_optical_flow(idx=0, category='STEMI', video_path='./Demos/DenseOpticalFlow', Compressed_path='./Data/Compressed', Processed_path='./Data/Processed'):
    '''This function writes a demo video for dense optical flow'''
    # create output directory
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    # read processed video
    cat_path_processed = os.path.join(Processed_path, category)
    filename = natsorted(os.listdir(cat_path_processed))[idx]
    video = read_video(os.path.join(cat_path_processed, filename))
    # read compressed video
    cat_path_compressed = os.path.join(Compressed_path, category)
    cvideo = read_video(os.path.join(cat_path_compressed, natsorted(os.listdir(cat_path_compressed))[idx]), color=True)
    # optical flow calculation
    xflow, yflow, mag, angle = dense_flow(video, bound=15, filename=filename)
    ang0 = standardize(np.mean(angle, axis=(1,2)))
    # angle interpolation
    t = np.arange(0,len(ang0))
    if len(ang0)>62:
        window_length = 31
    else:
        window_length = int(len(ang0)*0.5)
        if np.mod(window_length,2)==0:
            window_length+=1
    ang = savgol_filter(ang0, window_length, 2)
    localMaximas = list(argrelextrema(ang, np.greater, order=3)[0])
    localMaximasValues = [ang[i] for i in localMaximas]
    localMinimas = list(argrelextrema(ang, np.less, order=3)[0])
    localMinimasValues = [ang[i] for i in localMinimas]
    # end point calculation
    combined = localMaximas+localMinimas
    maxlocal = np.max(combined)
    if maxlocal in localMinimas:
        localMaximas.append(len(ang)-1)
        localMaximasValues.append(ang[-1])
    else:
        localMinimas.append(len(ang)-1)
        localMinimasValues.append(ang[-1])
    # plots
    row, column, fontsize = 2, 3, 20
    for l in tqdm(range(len(video))):
        plt.figure(figsize=(10,8))
        plt.subplot(row, column, (1,4))
        plt.imshow(cvideo[l])
        plt.title(f'Frame = {l:2d}', fontsize=fontsize-5)
        plt.axis('off')
        plt.subplot(row, column, 2)
        plt.plot(t, ang0, c='C0', linestyle='dotted', label='original')
        plt.plot(t, ang, c='C1', linewidth=2, label='smoothed')
        plt.scatter(localMaximas, localMaximasValues, s=1e2, c='C2', linewidths=3, marker='o', label='local maximas')
        plt.scatter(localMinimas, localMinimasValues, s=1e2, c='C3', linewidths=3, marker='o', label='local minimas')
        plt.scatter(l, ang[l], s=1e3, c='C4', linewidths=5, marker='|', label=f'Angle={ang[l]:0.02f}')
        plt.legend(fontsize=fontsize-10)
        plt.axis('off')
        plt.subplot(row, column, 3)
        plt.imshow(mag[l], cmap='gray')
        plt.title('Magnitudinal flow', fontsize=fontsize-5)
        plt.axis('off')
        plt.subplot(row, column, 5)
        plt.imshow(xflow[l], cmap='gray')
        plt.title('Radial flow', fontsize=fontsize-5)
        plt.axis('off')
        plt.subplot(row, column, 6)
        plt.imshow(yflow[l], cmap='gray')
        plt.title('Longitudinal flow', fontsize=fontsize-5)
        plt.axis('off')
        plt.tight_layout()
        plt.suptitle(filename, fontsize=fontsize)
        plt.savefig(os.path.join(video_path,f'image_{l:03d}.png'), bbox_inches='tight')
        plt.close()
    # video write
    plot2video(video_path, os.path.join(video_path,category+'_'+filename), fps=50)
    # remove plots
    images = [os.path.join(video_path,i) for i in os.listdir(video_path) if i.endswith('png')]
    for files in images:
        os.remove(files)
    print('\nVideo output: ', os.path.join(video_path,category+'_'+filename))
    
def ToImg(raw_flow, bound):
    '''This function standardize the video by enforcing bound'''
    flow = raw_flow
    flow [flow>bound] = bound
    flow [flow<-bound] = -bound
    flow -= -bound
    flow = (standardize(flow)*255).astype(np.uint8)
    return flow

def flows(allflows, bound):
    '''This function rescale video to 0~255 with the bound setting'''
    flow_x = ToImg(allflows[0][...,0], bound).astype(np.uint8)
    flow_y = ToImg(allflows[0][...,1], bound).astype(np.uint8)
    flow_mag = ToImg(allflows[1], bound).astype(np.uint8)
    flow_ang = (standardize(allflows[2])*255).astype(np.uint8)
    return flow_x, flow_y, flow_mag, flow_ang

def dense_flow(video, filename=['the given video'], bound=15, pyr_scale=0.5, levels=1, winsize=21, iterations=3, poly_n=5, poly_sigma=1.1, flags=0):
    ''' This function calculate dense optical flow given a graysacle video'''
    len_frame=len(video)
    frame_num=0
    gray, prev_gray = None, None
    xflows, yflows, magflows, angleflows = [], [], [], []
    while True:
        # last frame check
        if frame_num>=len_frame:
            break
        # initial frame check
        if frame_num==0:
            prev_gray = video[frame_num]
            frame_num+=1
        else:
            # dense flow magnitude and angle calculation
            gray = video[frame_num]
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        
            allflows = [flow, magnitude, angle]
            allflows = flows(allflows, bound)
            
            # magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            # angle = cv2.normalize(angle * 180 / np.pi / 2, None, 0, 255, cv2.NORM_MINMAX)
            # flow0 = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
            # flow1 = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
            # allflows = [flow0, flow1, magnitude, angle]
            
            # append the results for each frame
            if frame_num==1:
                xflows.append(allflows[0])
                yflows.append(allflows[1])
                magflows.append(allflows[2])
                angleflows.append(allflows[3])
            xflows.append(allflows[0])
            yflows.append(allflows[1])
            magflows.append(allflows[2])
            angleflows.append(allflows[3])
            # re-assign frames
            prev_gray = gray
            frame_num+=1
    # output files
    xflows = np.array(xflows).astype(np.uint8)
    yflows = np.array(yflows).astype(np.uint8)
    magflows = np.array(magflows).astype(np.uint8)
    angleflows = np.array(angleflows).astype(np.uint8)
    print('\nOptical flow calculation done for '+filename)
    return xflows, yflows, magflows, angleflows

def Distance_mag(y1,y2):
    dist = np.sqrt((y1[:,0,0]-y2[:,0,0])**2+(y1[:,0,1]-y2[:,0,1])**2)
    mean_dist = np.mean(dist)
    return mean_dist

def Distance_lon(y1,y2):
    dist = np.sqrt((y1[:,0,1]-y2[:,0,1])**2)
    mean_dist = np.mean(dist)
    return mean_dist

def Distance_rad(y1,y2):
    dist = np.sqrt((y1[:,0,0]-y2[:,0,0])**2)
    mean_dist = np.mean(dist)
    return mean_dist

class CoordinateStore:
    def __init__(self, old_frame, window_Name0):
        self.points = []
        self.old_frame = old_frame
        self.window_Name0 = window_Name0

    def select_point(self,event,x,y,flags,param):
        global counter
        global change
        global num
        # color_s = np.random.randint(0,255,(100,3))
        color_s = np.array([[0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 255], [30, 100, 255], [247, 134, 79], [182, 81, 156], [255, 255, 0]])
        if event == cv2.EVENT_LBUTTONDOWN:
            if np.mod(counter,num)==0:
                counter=1
                change+=1
            cv2.circle(self.old_frame,(x,y),2,color_s[change].tolist(),-1)
            cv2.imshow(self.window_Name0, self.old_frame)
            self.points.append((x,y))
            counter+=1

def sparse_flow(videopath, ws=75, mlvl=5):
    # read video
    cap = cv2.VideoCapture(videopath)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    global counter
    global change
    global num
    counter, change, num = 1, 0, 1
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (ws, ws),
                      maxLevel = mlvl,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Create some random colors
    color = np.array([[0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 255], [30, 100, 255], [247, 134, 79], [182, 81, 156], [255, 255, 0]])
    
    # Take first frame and find pixels in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    window_Name0 = 'Feature Point Selection'
    coordinateStore1 = CoordinateStore(old_frame, window_Name0)
    cv2.namedWindow(window_Name0, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_Name0, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_Name0, old_frame)
    # cv2.setMouseCallback('image', click_event)
    cv2.setMouseCallback(window_Name0, coordinateStore1.select_point)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    x_cord, y_cord = [],[]
    for i in coordinateStore1.points:
        x_cord.append(i[0])
        y_cord.append(i[1])
    
    p0 = np.zeros((len(x_cord),1,2),dtype=np.float32)
    for count in range(len(x_cord)):
        p0[count,0,0]=x_cord[count]
        p0[count,0,1]=y_cord[count]
        
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    # declare placeholders for pixel displacements
    mags, lons, rads = [], [], []
    mags.append(0)
    lons.append(0)
    rads.append(0)
    
    # iterator
    count=1
    while(ret==True):
        ret, frame = cap.read()
        if ret==False:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # displacement calculation
        dist_mag = Distance_mag(p0, p1)
        dist_lon = Distance_lon(p0, p1)
        dist_rad = Distance_rad(p0, p1)
        
        # record displacements
        mags.append(dist_mag)
        lons.append(dist_lon)
        rads.append(dist_rad)
                
        count += 1
            
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            frame = cv2.circle(frame,(int(a),int(b)),2,color[i].tolist(),-1)
            mask = np.zeros_like(frame)
        img = cv2.add(frame,mask)
        
        window_Name1 = 'Motion Tracking'
        cv2.namedWindow(window_Name1, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_Name1, cv2.WND_PROP_FULLSCREEN, 
                              cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_Name1,img)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
    
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    
    cv2.destroyAllWindows()
    cap.release()
    return (mags, lons, rads)

def demo_sparse_optical_flow(idx=0, category='STEMI', ws=75, mlvl=5, numpy_path='./PixelDisplacements', Processed_path='./Data/SingleCycleProcessed'):
    '''This function saves pixel displacements for sparse optical flow'''
    # create output directory
    if not os.path.exists(numpy_path):
        os.makedirs(numpy_path)
    # filepath
    cat_path = os.path.join(Processed_path, category)
    filename = natsorted(os.listdir(cat_path))[idx]
    filepath = os.path.join(cat_path, filename)
    # optical flow calculation
    allflow, xflow, yflow = sparse_flow(filepath, ws=ws, mlvl=mlvl)
    # plots
    plt.figure()
    plt.plot(allflow, color='C0', label='magnitudinal')
    plt.plot(xflow, color='C1', label='longitudinal')
    plt.plot(yflow, color='C2', label='radial')
    plt.legend()
    return allflow, xflow, yflow

def calculate_dense_optical_flows(input_path, output_path, BOUND=15):
    '''This function calculates dense optical flows and write them to the output path'''
    print('\n\n-----------------------------------')
    print('Optical flow calculation')
    print('-----------------------------------\n')
    # check for output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # list all categories
    category_names = natsorted(os.listdir(input_path))
    category_list = [os.path.join(input_path,i) for i in category_names]
    for n in range(len(category_list)):
        input_filenames = [i for i in natsorted(os.listdir(category_list[n]))]
        # output_directories
        output_directory = os.path.join(output_path, natsorted(os.listdir(input_path))[n])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # output directories with filenames for each segment
        output_segments = ['xFlows', 'yFlows', 'magFlows', 'angles']
        for seg in output_segments:
            segpath = os.path.join(output_directory, seg)
            if not os.path.exists(segpath):
                os.makedirs(segpath)
        # Check for missing files
        if len(os.listdir(os.path.join(output_directory, output_segments[0]))) == len(input_filenames):
            print('All '+category_names[n]+' files are already processed for optical flow calculation...')
        else:
            outputs = []
            for seg in output_segments:
                segpath = os.path.join(output_directory, seg)
                tag = '_'+seg+'.mp4'
                output_filenames = natsorted([os.path.join(segpath, os.path.splitext(i)[0]+tag) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(segpath)])
                outputs.append(output_filenames)
            input_filenames = natsorted([os.path.join(category_list[n], i) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(segpath)])
            print(category_names[n], f'[{len(output_filenames)} files to be processed]')
            print('-----------------------------------')
            print(input_filenames)
            print('-----------------------------------')
            # optical flow calculation
            for files in tqdm(range(len(input_filenames))):
                filename = input_filenames[files]
                video = read_video(filename)
                size = (video.shape[2], video.shape[1])
                allflows = dense_flow(video, filename=filename, bound=BOUND)
                # video write
                for nflow in range(len(allflows)):
                    write_video(allflows[nflow], size, fps=30, filepath=outputs[nflow][files])
                    
def calculate_sparse_optical_flows(input_path, output_path, ws=75, mlvl=5):
    '''This function calculates dense optical flows and write them to the output path'''
    print('\n\n-----------------------------------')
    print('Optical flow calculation')
    print('-----------------------------------\n')
    # check for output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # list all categories
    category_names = natsorted(os.listdir(input_path))
    category_list = [os.path.join(input_path,i) for i in category_names]
    for n in range(len(category_list)):
        input_filenames = [i for i in natsorted(os.listdir(category_list[n]))]
        # output_directories
        output_directory = os.path.join(output_path, natsorted(os.listdir(input_path))[n])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # output directories with filenames for each segment
        output_segments = ['magFlows', 'xFlows', 'yFlows']
        for seg in output_segments:
            segpath = os.path.join(output_directory, seg)
            if not os.path.exists(segpath):
                os.makedirs(segpath)
        # Check for missing files
        if len(os.listdir(os.path.join(output_directory, output_segments[0]))) == len(input_filenames):
            print('All '+category_names[n]+' files are already processed for optical flow calculation...')
        else:
            outputs = []
            for seg in output_segments:
                segpath = os.path.join(output_directory, seg)
                tag = '_'+seg+'.npy'
                output_filenames = natsorted([os.path.join(segpath, os.path.splitext(i)[0]+tag) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(segpath)])
                outputs.append(output_filenames)
            input_filenames = natsorted([os.path.join(category_list[n], i) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(segpath)])
            print(category_names[n], f'[{len(output_filenames)} files to be processed]')
            print('-----------------------------------')
            print(input_filenames)
            print('-----------------------------------')
            # optical flow calculation
            for files in tqdm(range(len(input_filenames))):
                filename = input_filenames[files]
                allflows = sparse_flow(filename, ws=ws, mlvl=mlvl)
                # numpy write
                for nflow in range(len(allflows)):
                    np.save(outputs[nflow][files], np.array(allflows[nflow]))
                
#%% Single cycle extraction
def demo_cycle_extraction(idx=0, category='STEMI', fps=30, video_path='./Data/Compressed', output_path='./Demos/CycleExtraction'):
    '''This function extracts a single cardiac cycle and writes a video for demo purpose'''
    # create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    video_path_category = os.path.join(video_path, category)
    filename = os.listdir(video_path_category)[idx]
    # read video
    video = read_video(os.path.join(video_path_category, filename), color=True)
    # find start and end frame
    st, en = extractCycle(video, filename)
    # extract cycle
    video = video[st:en]
    # video write
    savepath = os.path.join(output_path,category+'_'+filename)
    write_video(video, (video.shape[2], video.shape[1]), fps, savepath, color=True)
    print('\nVideo output: ', savepath)
    print(f'\nStart frame={st:02d} \nEnd frame={en:02d}')
    
def extractCycle(video, filename=[''], start_frame=False):
    '''This function extracts a single cardiac cycle using interactive plot'''
    import IPython
    import matplotlib
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt5')    
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    # Interactive plot
    fig, ax = plt.subplots()
    ax.imshow(video[0], cmap='gray')
    ax.set_title(filename)
    ax.axis('off')
    # Slider for start frame
    ax_sliderST = plt.axes([0.20, 0.07, 0.55, 0.03])
    if start_frame==True:
        sliderST = Slider(ax_sliderST, 'Start', 0, len(video), valinit=int(len(video)*0.4), color='steelblue')
    else:
        sliderST = Slider(ax_sliderST, 'Start', 0, len(video), valinit=0, color='steelblue')
    sliderST.valtext.set_text('SliceNo-{}'.format(int(0)))
    # Slider for end frame
    ax_sliderEN = plt.axes([0.20, 0.01, 0.55, 0.03], facecolor='steelblue')
    sliderEN = Slider(ax_sliderEN, 'End', 0, len(video), valinit=0, color='w')
    sliderEN.valtext.set_text('SliceNo-{}'.format(int(0)))
    # Update slider value
    def update(value):
        nslice = int(value)
        if video.ndim==4:
            ax.imshow(video[nslice])
        else:
            ax.imshow(video[nslice], cmap='gray')
        ax.axis('off')
        sliderST.valtext.set_text('SliceNo-{}'.format(int(sliderST.val)))
        sliderEN.valtext.set_text('SliceNo-{}'.format(int(sliderEN.val)))
        fig.canvas.draw_idle()
        return nslice
    sliderST.on_changed(update)
    sliderEN.on_changed(update)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show(block=True)
    shell.enable_matplotlib(gui='inline')
    # start and frame
    startFrame, endFrame, totalFrame = int(sliderST.val), int(sliderEN.val), len(video)
    return startFrame, endFrame, totalFrame

def single_cycle_extraction(input_path, output_path, automatic=True, cycle='full', 
                            numpy=True, st_frame=False, color=False, 
                            interpolation=False, frame=32):
    '''This function extracts single cardicac cycles given a video directory'''
    print('\n\n-----------------------------------')
    print('Single cycle extraction')
    print('-----------------------------------\n')
    # check for output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # list all categories
    category_names = natsorted(os.listdir(input_path))[::-1]
    category_list = [os.path.join(input_path,i) for i in category_names]
    for n in range(len(category_list)):
        input_filenames = [i for i in natsorted(os.listdir(category_list[n]))]
        output_directory = os.path.join(output_path, category_names[n])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Check for missing files
        if len(os.listdir(output_directory)) == len(input_filenames):
            print('All '+category_names[n]+' cycles are already extracted...')
        else:
            tag = '_singleCycle.mp4'
            outputs = natsorted([os.path.join(output_directory, os.path.splitext(i)[0]+tag) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            inputs = natsorted([os.path.join(category_list[n], i) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            input_filenames = natsorted([i for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            print(category_names[n], f'[{len(outputs)} files to be processed]')
            print('-----------------------------------')
            print(input_filenames)
            print('-----------------------------------')
            # Extract cycles
            for files in tqdm(range(len(inputs))):
                filename = input_filenames[files]
                video = read_video(inputs[files], color=color)
                if automatic == False:
                    start_frame, end_frame, total_frame = extractCycle(video, filename=filename, start_frame=st_frame)
                else:
                    start_frame, end_frame, total_frame = local_minima_optical_flow(video, filename=filename, cycle=cycle)
                if numpy==True:
                    numpy_path = './Numpys/'+category_names[n]
                    if not os.path.exists(numpy_path):
                        os.makedirs(numpy_path)
                    numpy_array = np.array([start_frame, end_frame, total_frame])
                    numpy_save_path = os.path.join(numpy_path, filename[0:8]+'.npy')
                    np.save(numpy_save_path, numpy_array)
                size = (video.shape[2], video.shape[1])
                extracted_video = video[start_frame: end_frame+1]
                if interpolation==True:
                    extracted_video = video_interpolation(extracted_video, frame=frame, color=color)
                # video write
                write_video(extracted_video, size, fps=30, filepath=outputs[files], color=color)
                
#%% Single cycle interpolation
def video_interpolation(video, frame=32, color=False):
    '''This function interpolates the total frames of a video to a given frame number'''
    if color==True:
        videoCh = []
        for ch in range(video.shape[-1]):
            videoCh.append(zoom(video[...,ch], (frame/video.shape[0], 1, 1)))
        video = np.stack(videoCh, axis=-1)
    else:
        video = zoom(video, (frame/video.shape[0], 1, 1))
    video[video<0]=0
    video = (standardize(video)*255).astype(np.uint8)
    return video

def single_cycle_interpolation(input_path, output_path, FRAME=32, color=False):
    '''This function interpolates the length of extracted single cardiac cycles to a given frame length'''
    print('\n\n-----------------------------------')
    print('Single cycle interpolation')
    print('-----------------------------------\n')
    # check for output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # list all categories
    category_names = natsorted(os.listdir(input_path))
    category_list = [os.path.join(input_path,i) for i in category_names]
    for n in range(len(category_list)):
        input_filenames = [i for i in natsorted(os.listdir(category_list[n]))]
        output_directory = os.path.join(output_path, natsorted(os.listdir(input_path))[n])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Check for missing files
        if len(os.listdir(output_directory)) == len(input_filenames):
            print('All '+category_names[n]+' cycles are already interpolated...')
        else:
            tag = '_interpolated.mp4'
            outputs = natsorted([os.path.join(output_directory, os.path.splitext(i)[0]+tag) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            inputs = natsorted([os.path.join(category_list[n], i) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            print(category_names[n], f'[{len(outputs)} files to be processed]')
            print('-----------------------------------')
            print(input_filenames)
            print('-----------------------------------')
            # Interpolate videos
            for files in tqdm(range(len(inputs))):
                video = read_video(inputs[files], color=color)
                size = (video.shape[2], video.shape[1])
                interpolated_video = video_interpolation(video, frame=FRAME, color=color)
                # video write
                write_video(interpolated_video, size, fps=30, filepath=outputs[files], color=color)

#%% ROI selection
def demo_select_ROI(idx=0, category='STEMI', fps=30, video_path='./Data/SingleCycleInterpolatedManual', output_path='./Demos/SelectROI'):
    '''This function selects and writes an ROI for demo purpose'''
    # create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    video_path_category = os.path.join(video_path, category)
    filename = os.listdir(video_path_category)[idx]
    # read video
    video = read_video(os.path.join(video_path_category, filename))
    # find ROI
    video = roiSelect(video, HEIGHT=128, WIDTH=160, filename=filename)
    # video write
    savepath = os.path.join(output_path,category+'_'+filename)
    write_video(video, (video.shape[2], video.shape[1]), fps, savepath)
    print('\nVideo output: ', savepath)
    
def roiSelect(video, HEIGHT=128, WIDTH=160, filename=['']):
    '''This function selects ROI interactively given a video'''
    # set window
    window_Name = filename+' [ROI Selection]'
    cv2.namedWindow(window_Name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_Name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # select ROI
    nslice = int(len(video)/2)
    roi_coord = cv2.selectROI(window_Name, video[nslice])
    # cv2.waitKey(0);
    cv2.destroyAllWindows()
    ROI = video[:, int(roi_coord[1]):int(roi_coord[1]+roi_coord[3]), int(roi_coord[0]):int(roi_coord[0]+roi_coord[2])]
    H = ROI.shape[1]
    ROI = zoom(ROI, (1, HEIGHT/H, HEIGHT/H))
    ROI[ROI<0] = 0
    W = ROI.shape[2]
    if W<WIDTH:
        st = int((WIDTH-W)/2)
        svideo = np.zeros((len(ROI), HEIGHT, WIDTH))
        svideo[:, :, st:st+W] = ROI[:]
    else:
        st = int((W-WIDTH)/2)
        svideo = ROI[:, :, st:st+WIDTH]
    svideo = (standardize(svideo)*255).astype(np.uint8)
    return svideo

def select_ROI(input_path, output_path, HEIGHT=128, WIDTH=160):
    '''This function interpolates the length of extracted single cardiac cycles to a given frame length'''
    print('\n\n-----------------------------------')
    print('Single cycle interpolation')
    print('-----------------------------------\n')
    # check for output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # list all categories
    category_names = natsorted(os.listdir(input_path))
    category_list = [os.path.join(input_path,i) for i in category_names]
    for n in range(len(category_list)):
        input_filenames = [i for i in natsorted(os.listdir(category_list[n]))]
        output_directory = os.path.join(output_path, natsorted(os.listdir(input_path))[n])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Check for missing files
        if len(os.listdir(output_directory)) == len(input_filenames):
            print('All '+category_names[n]+' ROI are extracted...')
        else:
            tag = '_ROI.mp4'
            outputs = natsorted([os.path.join(output_directory, os.path.splitext(i)[0]+tag) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            inputs = natsorted([os.path.join(category_list[n], i) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            input_filenames = natsorted([i for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            print(category_names[n], f'[{len(outputs)} files to be processed]')
            print('-----------------------------------')
            print(input_filenames)
            print('-----------------------------------')
            # Interpolate videos
            for files in tqdm(range(len(inputs))):
                video = read_video(inputs[files])
                ROI = roiSelect(video, HEIGHT, WIDTH, filename=input_filenames[files])
                # video write
                write_video(ROI, (WIDTH, HEIGHT), fps=30, filepath=outputs[files])
                
#%% data loader
def split3Dto2D(imgs, labels):
    '''This function each 3D frame as individual input'''
    imgs_2D, labels_2D = [],[]
    for n in tqdm(range(len(imgs))):
        d = imgs[n].shape[0]
        labels_2D.append(np.repeat(labels[n], d))
    imgs_2D = np.concatenate(imgs).astype(np.float32)
    labels_2D = np.concatenate(labels_2D).astype(np.uint8)
    return imgs_2D, labels_2D
    
def data_split(imgs, labels, frame=16):
    '''This function splits different length videos to given number of frames
    from the start and end of the videos'''
    split_imgs, split_labels = [], []
    for n in tqdm(range(len(imgs))):
        d, h, w = imgs[n].shape
        # check if video length smaller than the set frame
        if d<frame:
            split_img1 = np.zeros((frame, h, w), dtype=np.float32)
            split_img2 = np.zeros_like(split_img1)
            split_img1[0:d,:,:] = imgs[n][:]
            split_img2[frame-d:frame,:,:] = imgs[n][:]
        else:
            split_img1 = imgs[n][0:frame]
            split_img2 = imgs[n][d-frame:d]
        split_imgs.append(split_img1)
        split_imgs.append(split_img2)
        split_labels.append(labels[n])
        split_labels.append(labels[n])
    # declare arrays
    split_imgs = np.array(split_imgs, dtype=np.float32)
    split_labels = np.array(split_labels, dtype=np.uint8)
    return split_imgs, split_labels
    
    
def data_loader(input_path='./Data/Processed', data_type='3D', frame=16):
    '''This function loads videos from given directory'''
    # declare placeholders
    imgs, labels, lengths = [], [], []
    # list of all categories
    input_category = natsorted(os.listdir(input_path))
    # load videos for each category
    for n in range(len(input_category)):
        category_name = input_category[n]
        category_location = os.path.join(input_path, category_name)
        filelist = [os.path.join(category_location, i) for i in natsorted(os.listdir(category_location))]
        print('\n-----------------------------------')
        print(f'Total {len(filelist):d} '+category_name+' files are found...\n')
        for files in tqdm(range(len(filelist))):
            img = standardize(read_video(filelist[files]))
            imgs.append(img)
            lengths.append(len(img))
            labels.append(n)
        print('\nAll '+category_name+' files are loaded...')
    # check data type
    if data_type=='2D':
        imgs, labels = split3Dto2D(imgs, labels)
        print(f'\nTotal number of frames = {len(labels):d}\n')
    else:
        # check for video length
        if len(np.unique(lengths))>1:
            print('\n-----------------------------------')
            print(f'Length of the videos are different, splitting into length of {frame:d} frames...\n')
            imgs, labels = data_split(imgs, labels, frame=frame)
        # declare arrays
        imgs = np.array(imgs, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)
        print(f'\nTotal number of videos = {len(labels):d}\n')
    return imgs, labels

#%% data write
def write_NIFTI(input_path, output_path):
    '''This function write niftis in the video directory'''
    print('\n\n-----------------------------------')
    print('NFITI writing')
    print('-----------------------------------\n')
    # check for output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # list all categories
    category_names = natsorted(os.listdir(input_path))
    category_list = [os.path.join(input_path,i) for i in category_names]
    for n in range(len(category_list)):
        input_filenames = [i for i in natsorted(os.listdir(category_list[n]))]
        output_directory = os.path.join(output_path, natsorted(os.listdir(input_path))[n])
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Check for missing files
        if len(os.listdir(output_directory)) == len(input_filenames):
            print('All '+category_names[n]+' files are already written...')
        else:
            tag = '_niftis.nii.gz'
            output_filenames = natsorted([os.path.join(output_directory, os.path.splitext(i)[0]+tag) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            input_filenames = natsorted([os.path.join(category_list[n], i) for i in input_filenames if os.path.splitext(i)[0]+tag not in os.listdir(output_directory)])
            print(category_names[n], f'[{len(output_filenames)} files to be processed]')
            print('-----------------------------------')
            print(input_filenames)
            print('-----------------------------------')
            # Compress videos
            for files in tqdm(range(len(input_filenames))):
                video = read_video(input_filenames[files], color=False)
                write_nifti(video, output_filenames[files])

#%% model training and inference
def get_Kfold(imgs, labels, fold, n_splits=5, validation_split=0.1, random_state=None, shuffle=False):
    '''This function generate train and test samples for each K-fold'''
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    for i, (trn_index, tst_index) in enumerate(skf.split(imgs, labels)):
        if i==fold:
            train_index, test_index = trn_index, tst_index
    xtrain, xtest, ytrain, ytest = imgs[train_index], imgs[test_index], labels[train_index], labels[test_index]
    if validation_split != False:
        xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=validation_split, random_state=0, stratify=ytrain)
    else:
        xval, yval = xtest, ytest
    print(f'\nNumber of trainData={len(xtrain):d}, validationData={len(xval):d}, testData={len(xtest):d}')
    return xtrain, xval, xtest, ytrain, yval, ytest

def save_parameters(filename, ilr, loss, architecture_type=[''], model_type=[''], temporal_type=[''], parameters=['']):
    '''This function save model parameters to a text file'''
    with open(filename, 'w') as file: 
        books = ['Model Directory = '+os.path.splitext(filename)[0]+'\n',
                 '---------------------------------------\n',
                 'Architecture_type = '+architecture_type+'\n',
                 'Temporal Type = '+temporal_type+'\n',
                 'Model Type = '+model_type+'\n',
                 '---------------------------------------\n',
                 'Initial Learning Rate = '+parameters[0]+'\n',
                 'Loss Function = '+loss+'\n',
                 'Batch Normalization = '+parameters[1]+'\n',
                 'Dropout Rate = '+parameters[2]+'\n',
                 '---------------------------------------\n',
                 'Convolution Layers = '+parameters[3]+'\n',
                 'Recurrent Layers = '+parameters[4]+'\n',
                 'Fully Connected Layers = '+parameters[5]+'\n']
        file.writelines("% s" % data for data in books)

def model_train_inference(inputs, labels, model,
                          architecture_type, model_type, ilr, loss, temporal_type='False',
                          checkpoint='./savedModels', tag='version1',
                          total_split=5, validation_split=0.1, epoch=25, batch_size=4,
                          train=False, random_state=None, shuffle=False,
                          model_parameters=[]):
    '''This function train or test the model'''
    # model directory
    if architecture_type=='CNN':
        save_path = os.path.join(checkpoint, architecture_type, temporal_type, model_type)
    else:
        save_path = os.path.join(checkpoint, architecture_type, model_type)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save parameters
    if train == True:
        parameter_path = os.path.join(save_path, tag+'.txt')
        save_parameters(filename=parameter_path, ilr=ilr, loss=loss, 
                        architecture_type=architecture_type, model_type=model_type, temporal_type=temporal_type,
                        parameters=model_parameters)
        print('\n-----------------------------------')
        print('\nModel parameter path ->', parameter_path, '\n')
    # placeholders
    xtests, ytests, prediction = [], [], []
    # K-fold cross validation
    for fold in tqdm(range(total_split)):
        print('\n-----------------------------------')
        print(f'Fold-{fold:02d}')
        # train, validation and test data
        xtrain, xval, xtest, ytrain, yval, ytest = get_Kfold(inputs, labels, fold, n_splits=total_split,
                                                             validation_split=validation_split, random_state=random_state, shuffle=shuffle)
        # model name
        chkpoint_name = os.path.join(save_path, tag+f'_fold-{fold:02d}.hdf5')
        print('\nModel path ->',chkpoint_name, '\n')
        # check for train or test
        if train==True:
            print(f'Training fold {fold:02d}.....\n')
            cp = ModelCheckpoint(chkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=ilr, decay_steps=1000, decay_rate=0.9)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, epsilon=1e-07), 
                          loss=loss, metrics=['accuracy'])
            # save initial weights
            initial_weight_path = os.path.join(save_path, tag+'_initialWeights.hdf5')
            if fold==0:
                model.save_weights(initial_weight_path)
            model.load_weights(initial_weight_path)
            # train the model
            model.fit(xtrain, ytrain, 
                      epochs=epoch, 
                      batch_size=batch_size, 
                      validation_data=(xval, yval),
                      callbacks=cp, verbose=1)
        else:
            # model prediction on test data
            if not os.path.exists(chkpoint_name):
                raise Exception('The model is not trained yet, please train the model first...')                
        # model prediction on test data
        model.load_weights(chkpoint_name)
        print(f'\nTesting fold {fold:02d}.....\n')
        prediction.append(model.predict(xtest, batch_size=batch_size, verbose=1))
        xtests.append(xtest)
        ytests.append(ytest)
    return xtests, ytests, prediction

def get_results(y_true, y_pred):
    '''This function calculates result metrics'''
    accuracy, sensitivity, specificity, precision = [], [], [], []
    for fold in tqdm(range(len(y_pred))):
        ytrue = 1-y_true[fold]
        ypred = 1-np.argmax(y_pred[fold], axis=-1)
        tn, fp, fn, tp = confusion_matrix(ytrue, ypred).ravel()
        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))
        precision.append(tp / (tp + fp))
        accuracy.append(accuracy_score(ytrue, ypred))
    print('\n\n-----------------------------------')
    print(f'Accuracy = {np.mean(accuracy):0.02f}')
    print(f'Sensitivity = {np.mean(sensitivity):0.02f}')
    print(f'Specificity = {np.mean(specificity):0.02f}')
    print(f'Precision = {np.mean(precision):0.02f}')
    print('-----------------------------------\n')
    return np.mean(accuracy), np.mean(sensitivity), np.mean(specificity), np.mean(precision)

#%% Diastolic disfunction
def curve_fit(Y, k=[5,2]):
    y = savgol_filter(Y, k[0], k[1], mode='nearest')
    return y

def numpy_load(data_path, plot=False):
    category_list = natsorted(os.listdir(data_path))
    input_path = [os.path.join(data_path, i) for i in category_list]
    cat_lengths, cat_dias, filelists = [], [], []
    for n in range(len(category_list)):
        diastole, relative_length, names = [], [], []
        filenames = natsorted(os.listdir(input_path[n]))
        input_files = [os.path.join(input_path[n],i) for i in filenames]
        for files in range(len(input_files)):
            data = np.load(input_files[files])
            if data[0] != 0:
                length = (data[1]-data[0])/(data[2]-data[0])
                dias = (data[2]-data[0])/data[2]
                relative_length.append(length)
                diastole.append(dias)
                names.append(filenames[files])
        cat_lengths.append(relative_length)
        cat_dias.append(diastole)
        filelists.append(names)
        
        # print('\n',category_list[n], np.around(relative_length, decimals=2))
        mean, std = np.around(np.mean(relative_length), decimals=2), np.around(np.std(relative_length), decimals=2)
        dmean, dstd = np.around(np.mean(diastole), decimals=2), np.around(np.std(diastole), decimals=2)
        print('\n['+category_list[n]+f'] Dias.: {dmean:0.02f}'+u' \u00B1 '+f'{dstd:0.02f}'+f'\t\tRelax.: {mean:0.02f}'+u' \u00B1 '+f'{std:0.02f} (n={len(relative_length):d})')
        print('------------------------------------------')
    
    if plot==True:
        plt.figure(figsize=(10,2))
        plt.hlines(0,0,1, color='k', alpha=0.5)
        # plt.vlines(threshold,-0.009,0.009, color='k', ls='dotted', label=f'Data Mean:{threshold:0.2f}')  # Draw a horizontal line
        plt.xlim(0,0.65)
        plt.xticks(np.arange(0,0.65,0.05))
        plt.ylim(-0.01,0.01)
        colors = ['tab:blue', 'tab:red', 'tab:green'][::-1]
        markers = ['*', 'o', 'd'][::-1]
        labels = [f'TTS (mean={np.mean(np.array(cat_lengths[2])):0.2f})', 
                  f'STEMI (mean={np.mean(np.array(cat_lengths[1])):0.2f})',
                  f'CTRL (mean={np.mean(np.array(cat_lengths[0])):0.2f})'][::-1]
        ms = [125, 100, 75]
        alphas = [0.5, 0.5, 0.5]
        for n in range(len(cat_lengths)):
            plt.scatter(cat_lengths[n], np.zeros_like(cat_lengths[n]), s=ms[n], color=colors[n], marker=markers[n], label=labels[n], alpha=alphas[n])
        plt.legend(fontsize=10, loc='lower right', ncol=len(cat_lengths)+1)
        plt.tight_layout()
    return filelists, cat_lengths

def density_plot_relax(datapath, thr=0.27, plot=True, pvalue=True, errors=True, lineplot=True):
    filelists, lengths = numpy_load(datapath, plot=lineplot)
    if errors==True:
        STEMI_MRN = [filelists[1][j]+' = '+str(np.around(lengths[1][j], decimals=2)) for j in [i for i in np.arange(0,len(lengths[1])) if lengths[1][i]>=thr]]
        TTS_MRN = [filelists[2][j]+' = '+str(np.around(lengths[2][j], decimals=2)) for j in [i for i in np.arange(0,len(lengths[2])) if lengths[2][i]<thr]]
        print('\n------------------------------------------')
        print('TTS')
        print(TTS_MRN, f' [n={len(TTS_MRN):d}]')
        print('\n------------------------------------------')
        print('STEMI')
        print(STEMI_MRN, f' [n={len(STEMI_MRN):d}]')
    
    if pvalue==True:
        print(f'\np-value (Relax.): {ttest_ind(lengths[1], lengths[2])[1]:0.03f}')
    
    if plot==True:
        labels = [f'TTS (mean={np.mean(np.array(lengths[2])):0.2f})', 
                  f'STEMI (mean={np.mean(np.array(lengths[1])):0.2f})',
                  f'CTRL (mean={np.mean(np.array(lengths[0])):0.2f})'][::-1]
        fs, lw = 20, 5
        plt.figure(figsize=(8,3))
        sns.kdeplot(lengths[0], bw_method=0.25, color='tab:green', ls='solid', lw=3, label=labels[0])
        sns.kdeplot(lengths[1], bw_method=0.25, color='tab:red', ls='solid', lw=3, label=labels[1])
        sns.kdeplot(lengths[2], bw_method=0.25, color='tab:blue', ls='solid', lw=3, label=labels[2])
        plt.xlim(0,1)
        plt.xticks(np.arange(0,1,0.1), fontsize=fs-7)
        plt.yticks(fontsize=fs-7)
        plt.legend(fontsize=fs-5, loc='upper right')
        plt.ylabel('Density', fontsize=fs)
        plt.xlabel('(Isovolumic Relaxation)/Diastole', fontsize=fs-5)
        plt.tight_layout()
        
def sparseFlow_displacement(data_path, diastolic_path):
    
    cats = natsorted(os.listdir(data_path))
    
    dataCat = [os.path.join(data_path,i) for i in cats]
    
    diasCat = [os.path.join(diastolic_path,i) for i in cats]
    
    catFlows, catDiasFlows, catRelaxFlows = [], [], []
    
    for n in range(len(cats)):
        
        flowCats = natsorted([os.path.join(dataCat[n],i) for i in natsorted(os.listdir(dataCat[n]))])
        
        match = [x for x in [i[0:8] for i in natsorted(os.listdir(diasCat[n]))] if x in [i[0:8] for i in natsorted(os.listdir(flowCats[0]))]]
        fileDiasCats = natsorted([os.path.join(diasCat[n],i) for i in natsorted(os.listdir(diasCat[n])) if i[0:8] in match])
                
        diasNumpys = []
        for files in fileDiasCats:
            es, er, ed = np.load(files)
            # if es!=0 or er!=1:
            #     diasNumpys.append([int((es*32)/ed), int((er*32)/ed)])
            diasNumpys.append([int((es*32)/ed), int((er*32)/ed)])
                
        
        fileFlowCats, allFlows, allDiasFlows, allRelaxFlows = [], [], [], []
        for f in range(len(flowCats)):
            fileFlowCats.append([os.path.join(flowCats[f],i) for i in natsorted(os.listdir(flowCats[f]))])
            flows, diasflows, relaxflows = [], [], []
            for files in fileFlowCats[f]:
                
                orgFlow = np.array(np.load(files))
                esFrame, erFrame = diasNumpys[f][0], diasNumpys[f][1]
                
                x = np.linspace(0,31,len(orgFlow))
                x_new = np.arange(0,32)
                interp = interpolate.interp1d(x, orgFlow)
                newFlow = interp(x_new)
                flows.append(newFlow)
                
                orgDiasFlow = newFlow[esFrame:]
                x = np.linspace(0,31,len(orgDiasFlow))
                x_new = np.arange(0,32)
                interp = interpolate.interp1d(x, orgDiasFlow)
                newDiasFlow = interp(x_new)
                diasflows.append(newDiasFlow)
                # if n==1 and f==0:
                #     plt.figure()
                #     plt.plot(x_new, newDiasFlow)
                
                orgRelaxFlow = newFlow[esFrame:erFrame]
                x = np.linspace(0,31,len(orgRelaxFlow))
                x_new = np.arange(0,32)
                interp = interpolate.interp1d(x, orgRelaxFlow)
                newRelaxFlow = interp(x_new)
                relaxflows.append(newRelaxFlow)
            
            allFlows.append(flows)
            allDiasFlows.append(diasflows)
            allRelaxFlows.append(relaxflows)
        
        catFlows.append(allFlows)
        catDiasFlows.append(allDiasFlows)
        catRelaxFlows.append(allRelaxFlows)               
                
    flowNames = ['Magnitudinal', 'Longitudinal', 'Radial']
    colors = ['tab:green', 'tab:red', 'tab:blue']
    r, c, lw, fs = 1, 3, 3, 25
    
    plt.figure(figsize=(15,4))
    for i in range(c):
        plt.subplot(r,c,i+1)
        for j in range(len(catFlows)):
            plt.plot(np.linspace(0,1,32), np.mean(np.array(catFlows[j][i]), axis=0), label=cats[j], c=colors[j], lw=lw)
        plt.xlim(0,1)
        plt.title(flowNames[i], fontsize=fs-5)
        plt.legend(loc='lower right', ncol=len(cats)+1, fontsize=fs-10)
    plt.suptitle('Cardiac Cycle', fontsize=fs)
    plt.tight_layout()
    
    plt.figure(figsize=(15,4))
    for i in range(c):
        plt.subplot(r,c,i+1)
        for j in range(len(catDiasFlows)):
            plt.plot(np.linspace(0,1,32), np.mean(np.array(catDiasFlows[j][i]), axis=0), label=cats[j], c=colors[j], lw=lw)
        plt.xlim(0,1)
        plt.title(flowNames[i], fontsize=fs-5)
        plt.legend(loc='upper left', ncol=len(cats)+1, fontsize=fs-10)
    plt.suptitle('Diastole', fontsize=fs)
    plt.tight_layout()
    
    plt.figure(figsize=(15,4))
    for i in range(c):
        plt.subplot(r,c,i+1)
        for j in range(len(catRelaxFlows)):
            plt.plot(np.linspace(0,1,32), np.mean(np.array(catRelaxFlows[j][i]), axis=0), label=cats[j], c=colors[j], lw=lw)
        plt.xlim(0,1)
        plt.title(flowNames[i], fontsize=fs-5)
        plt.legend(loc='upper left', ncol=len(cats)+1, fontsize=fs-10)
    plt.suptitle('IsoVolumic Relaxation Period', fontsize=fs)
    plt.tight_layout()  


def sparseFlow_displacement_interpolated(data_path, diastolic_path, plots=True, longplot=True, nplot=11, smoothing=[]):
    
    cats = natsorted(os.listdir(data_path))
    
    dataCat = [os.path.join(data_path,i) for i in cats]
    
    diasCat = [os.path.join(diastolic_path,i) for i in cats]
    
    catFlows, catDiasFlows, catRelaxFlows = [], [], []
    
    maxCat, maxDias, maxRelax = 0, 0, 0
    
    for n in range(len(cats)):
        
        flowCats = natsorted([os.path.join(dataCat[n],i) for i in natsorted(os.listdir(dataCat[n]))])
        
        match = [x for x in [i[0:8] for i in natsorted(os.listdir(diasCat[n]))] if x in [i[0:8] for i in natsorted(os.listdir(flowCats[0]))]]
        fileDiasCats = natsorted([os.path.join(diasCat[n],i) for i in natsorted(os.listdir(diasCat[n])) if i[0:8] in match])
                
        diasNumpys = []
        for files in fileDiasCats:
            es, er, ed = np.load(files)
            diasNumpys.append([int((es*32)/ed), int((er*32)/ed)])
                
        
        fileFlowCats, allFlows, allDiasFlows, allRelaxFlows = [], [], [], []
        for f in range(len(flowCats)):
            fileFlowCats.append([os.path.join(flowCats[f],i) for i in natsorted(os.listdir(flowCats[f]))])
            flows, diasflows, relaxflows = [], [], []
            for files in fileFlowCats[f]:
                
                esFrame, erFrame = diasNumpys[f][0]-1, diasNumpys[f][1]-1
                
                orgFlow = np.array(np.load(files))[1:]
                # orgFlow = np.concatenate((np.expand_dims(orgFlow[0], axis=0), orgFlow), axis=0)
                flows.append(orgFlow)
                
                orgDiasFlow = orgFlow[esFrame:]
                x = np.linspace(0,30,len(orgDiasFlow))
                x_new = np.arange(0,31)
                interp = interpolate.interp1d(x, orgDiasFlow)
                newDiasFlow = interp(x_new)
                diasflows.append(newDiasFlow)
                
                orgRelaxFlow = orgFlow[esFrame:erFrame]
                x = np.linspace(0,30,len(orgRelaxFlow))
                x_new = np.arange(0,31)
                interp = interpolate.interp1d(x, orgRelaxFlow)
                newRelaxFlow = interp(x_new)
                relaxflows.append(newRelaxFlow)
            
            allFlows.append(flows)
            allDiasFlows.append(diasflows)
            allRelaxFlows.append(relaxflows)
            
        
        catFlows.append(allFlows)
        catDiasFlows.append(allDiasFlows)
        catRelaxFlows.append(allRelaxFlows) 
        
        mcF, mdF, mrF = np.max(np.mean(np.array(allFlows[0]), axis=0)), np.max(np.mean(np.array(allDiasFlows[0]), axis=0)), np.max(np.mean(np.array(allRelaxFlows[0]), axis=0))
        if mcF > maxCat:
            maxCat = mcF
        if mdF > maxDias:
            maxDias = mdF
        if mrF > maxRelax:
            maxRelax = mrF
    
    maxCat, maxDias, maxRelax = 1.1*maxCat, 1.1*maxDias, 1.1*maxRelax
                        
    flowNames = ['Magnitudinal', 'Longitudinal', 'Radial']
    colors = ['tab:green', 'tab:red', 'tab:blue']
    r, c, lw = 1, 3, 7
    fsT, fsST, fsL, fsXY = 40, 50, 30, 25
    intervals, dec = 5, 1
    
    if plots==True:
        plt.figure(figsize=(35,8))
        for i in range(c):
            plt.subplot(r,c,i+1)
            for j in range(len(catFlows)):
                x, y = np.linspace(0,1,31), np.mean(np.array(catFlows[j][i]), axis=0)
                if len(smoothing)==2:
                    y = curve_fit(y, smoothing)
                plt.plot(x, y, label=cats[j], c=colors[j], lw=lw)
            plt.xlim(0,1)
            plt.ylim(0,maxCat)
            plt.xticks(np.round(np.linspace(0,1,intervals), decimals=dec), fontsize=fsXY)
            plt.yticks(np.round(np.linspace(0,maxCat,intervals), decimals=dec), fontsize=fsXY)
            plt.title(flowNames[i], fontsize=fsT)
            plt.legend(loc='upper right', ncol=len(cats)+1, fontsize=fsL)
        plt.suptitle('Cardiac Cycle', fontsize=fsST)
        plt.tight_layout()
        
        plt.figure(figsize=(35,8))
        for i in range(c):
            plt.subplot(r,c,i+1)
            for j in range(len(catDiasFlows)):
                x, y = np.linspace(0,1,31), np.mean(np.array(catDiasFlows[j][i]), axis=0)
                if len(smoothing)==2:
                    y = curve_fit(y, smoothing)
                plt.plot(x, y, label=cats[j], c=colors[j], lw=lw)
            plt.xlim(0,1)
            plt.ylim(0,maxDias)
            plt.xticks(np.round(np.linspace(0,1,intervals), decimals=dec), fontsize=fsXY)
            plt.yticks(np.round(np.linspace(0,maxDias,intervals), decimals=dec), fontsize=fsXY)
            plt.title(flowNames[i], fontsize=fsT)
            plt.legend(loc='upper right', ncol=len(cats)+1, fontsize=fsL)
        plt.suptitle('Diastole', fontsize=fsST)
        plt.tight_layout()
        
        plt.figure(figsize=(35,8))
        for i in range(c):
            plt.subplot(r,c,i+1)
            for j in range(len(catRelaxFlows)):
                x, y = np.linspace(0,1,31), np.mean(np.array(catRelaxFlows[j][i]), axis=0)
                if len(smoothing)==2:
                    y = curve_fit(y, smoothing)
                plt.plot(x, y, label=cats[j], c=colors[j], lw=lw)
            plt.xlim(0,1)
            plt.ylim(0,maxRelax)
            plt.xticks(np.round(np.linspace(0,1,intervals), decimals=dec), fontsize=fsXY)
            plt.yticks(np.round(np.linspace(0,maxRelax,intervals), decimals=dec), fontsize=fsXY)
            plt.title(flowNames[i], fontsize=fsT)
            plt.legend(loc='upper left', ncol=len(cats)+1, fontsize=fsL)
        plt.suptitle('IsoVolumic Relaxation Period', fontsize=fsST)
        plt.tight_layout()
        
        flowTypes = ['Cardiac Cycle', 'Diastole', 'IsoVolumic Relaxation Period']
        locs = ['upper right', 'upper right', 'upper right']
        allFlows = [catFlows, catDiasFlows, catRelaxFlows]
        maxs = [maxCat, maxDias, maxRelax]
        plt.figure(figsize=(35,8))
        for i in range(len(allFlows)):
            plt.subplot(r,c,i+1)
            for j in range(len(cats)):
                x, y = np.linspace(0,1,31), np.mean(np.array(allFlows[i][j][0]), axis=0)
                if len(smoothing)==2:
                    y = curve_fit(y, smoothing)
                plt.plot(x, y, label=cats[j], c=colors[j], lw=lw)
                plt.xlim(0,1)
                plt.ylim(0,maxs[i])
                plt.xticks(np.round(np.linspace(0,1,intervals), decimals=dec), fontsize=fsXY)
                plt.yticks(np.round(np.linspace(0,maxs[0],intervals), decimals=dec), fontsize=fsXY)
            plt.title(flowTypes[i], fontsize=fsT)
            plt.legend(loc=locs[i], ncol=len(cats)+1, fontsize=fsL)
        plt.suptitle('Magnitudinal pixel displacement per frame', fontsize=fsST)
        plt.tight_layout()
    
    if longplot==True:
        fsL, fsXY, lw = 70, 70, 20
        yrange = np.round(np.linspace(0,1.2*np.max(np.mean(catFlows[1][0], axis=0)),5), decimals=dec)
        ytcks = [''] + [str(i) for i in yrange[1:]]
        plt.figure(figsize=(nplot*5,10))
        for n in range(1,3):
            x, y = np.linspace(0,1,31), np.mean(catFlows[n][0], axis=0)
            if len(smoothing)==2:
                y = curve_fit(y, smoothing)
            plt.plot(x, y, label=cats[n], c=colors[n], lw=lw)
        plt.xlim(0,1)
        plt.ylabel('Pixel displacement', fontsize=fsL-10)
        plt.xticks(np.round(np.linspace(0,1,11), decimals=dec), fontsize=fsXY)
        plt.yticks(yrange, ytcks, fontsize=fsXY)
        plt.legend(fontsize=fsL, loc='lower right', ncols=2)
        plt.title('1 cardiac cycle', fontsize=fsL+20)
        plt.tight_layout()
        
def motion_density_plot(filepathTTSflow, filepathSTEMIflow, nplot=9, smoothing=[5,3]):
    fTTS = read_video(filepathTTSflow, color=False)
    fMI = read_video(filepathSTEMIflow, color=False)

    infTTS = standardize(np.mean(fTTS, axis=(1,2)))
    infMI = standardize(np.mean(fMI, axis=(1,2)))

    fsL, fsXY, lw = 55, 55, 20
    
    infTTS = curve_fit(infTTS, smoothing)
    infMI = curve_fit(infMI, smoothing)

    catFlows = [infTTS, infMI]
    labels = ['TTS', 'STEMI']
    colors = ['tab:blue', 'tab:red']
    
    for n in range(len(catFlows)):
        yrange = np.linspace(0,1,5)
        ytcks = [''] + [str(i) for i in np.round(yrange[1:], decimals=1)]
        x, y = np.linspace(0,1,32), catFlows[n]
        plt.figure(figsize=(nplot*5,7))
        plt.plot(x, y, label=labels[n], c=colors[n], lw=lw)
        plt.xlim(0,1)
        # plt.ylabel('Heatmap Density', fontsize=fsL-5)
        plt.xticks(np.linspace(0,1,11), fontsize=fsXY)
        plt.yticks(yrange, ytcks, fontsize=fsXY)
        plt.legend(fontsize=fsL, loc='upper right', ncols=2)
        plt.title('Heatmap Intensity', fontsize=fsL)
        # plt.title('1 cardiac cycle', fontsize=fsL+20)
        plt.tight_layout()