#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:58:22 2024

@author: fazaman
"""

# import numpy as np
# import os
from utilities import tools, models

#%% load data
# video directory
data_path = './Data/SingleCycleInterpolatedAutomatic' # choices -> './Data/Processed' './Data/SingleCycleInterpolatedAutomatic', './Data/SingleCycleInterpolatedAutomaticROI'
# data dimension
data_type = '3D' # choices -> '3D', '2D'
# data load
datas, labels = tools.data_loader(input_path=data_path, data_type=data_type, frame=16)

#%% model parameters
# choose architecture
model_architecture = 'CNN' # choices -> 'CNN', 'RNN', 'CNN-RNN'
# choose model
model_type = 'ECM-paper' # choices -> ['CNN' -> 'VGG16', 'VGG19', 'ECM-paper', 'custom'], ['RNN'/'RNN-CNN' -> 'ECM-paper', 'custom']
# choose temporal axis convolution type
temporal_type = 'spatio-temporal' # choices -> 'spatio_temporal', 'temporal', 'temporal_fixed_depth'
# initial learning rate for exponential decay
initial_learning_rate = 1e-5
# initial filter/kernel number
filter_number = 16
# drop out rate
drop_out = 0.25
# batch normalization
batch_normalization = True
# number of layers for each convolution block
conv_layers = [3, 3, 3]
# number of nodes for each recurrent layer
rnn_layers = [1024, 512]
# number of nodes for each fully connected layer
dense_layers = [256,128]
# load models with reshaped data
X_train, y_train, model, parameters = models.load_model(inputs=datas, outputs=labels, initial_learning_rate=initial_learning_rate,
                                                        model_architecture=model_architecture, model_type=model_type, temporal_type=temporal_type,
                                                        filter_number=filter_number, drop_out=drop_out, batch_normalization=batch_normalization,
                                                        conv_layers=conv_layers, rnn_layers=rnn_layers, dense_layers=dense_layers, show_model=True)

#%% train/test hyperparameters
# set model train or test
TRAIN = False # choices -> True, False
# number of training epochs
EPOCH = 25
# number of batch size
BATCH_SIZE = 25
# save model
checkpoint_path = './savedModels'
# model tag
tag = 'version1'
save_tag = data_path.split('/')[-1]+'_'+tag
# Kfold cross validation
total_splits = 5
# validation data
validation_split = False
# initial learning rate for exponential decay
initial_learning_rate = 1e-2
# loss function
lossFunction = 'sparse_categorical_crossentropy'

#%% model train/inference
X_test, y_test, y_pred = tools.model_train_inference(X_train, y_train, model, architecture_type=model_architecture,
                                                     model_type=model_type, temporal_type=temporal_type,
                                                     ilr=initial_learning_rate, loss=lossFunction,
                                                     checkpoint=checkpoint_path, tag=save_tag,
                                                     total_split=total_splits, validation_split=validation_split, epoch=EPOCH, batch_size=BATCH_SIZE,
                                                     train=TRAIN, random_state=None, shuffle=False,
                                                     model_parameters=parameters)

#%% result plot
accuracy, sensitivity, specificity, precision = tools.get_results(y_test, y_pred)
