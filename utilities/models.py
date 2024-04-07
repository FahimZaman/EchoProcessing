#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:47:39 2024

@author: fazaman
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.callbacks import ModelCheckpoint
import numpy as np

#%% CNN models
def VGG_3D(input_shape=(16, 256, 352, 1), filter_number=16, num_labels=2, initial_learning_rate = 1e-5, 
        drop_out=False, batch_normalization=False,
        conv_layers=[2,2,3,3,3], dense_layers=[256, 128],
        data_type='3D', temporal_type='spatio_temporal'):
    '''This function builds a 2D/3D VGG network given data type, temporal axis type,
    number of convolution and dense layers'''
    # input layer
    inputs = layers.Input(shape=input_shape, dtype=tf.float32, name='video_input')
    # conv+pool layers
    for lyr in range(len(conv_layers)):
        for conv in range(conv_layers[lyr]):
            if lyr==0 and conv==0:
                if data_type=='3D':
                    if temporal_type=='spatio_temporal':
                        x = layers.Conv3D((lyr+1)*filter_number, 3, activation='relu', padding='same') (inputs)
                    else:
                        x = layers.Conv3D((lyr+1)*filter_number, (3, 1, 1) , activation='relu', padding='same') (inputs)
                else:
                    x = layers.Conv2D((lyr+1)*filter_number, 3, activation='relu', padding='same') (inputs)
            else:
                if data_type=='3D':
                    if temporal_type=='spatio_temporal':
                        x = layers.Conv3D((lyr+1)*filter_number, 3, activation='relu', padding='same') (x)
                    else:
                        x = layers.Conv3D((lyr+1)*filter_number, (3, 1, 1), activation='relu', padding='same') (x)
                else:
                    x = layers.Conv2D((lyr+1)*filter_number, 3, activation='relu', padding='same') (x)
            if batch_normalization != False:
                x = layers.BatchNormalization() (x)
            if drop_out != False:
                x = layers.Dropout(drop_out) (x)
        # pooling layer
        if lyr != len(conv_layers)-1:
            if data_type=='3D':
                if temporal_type=='temporal_fixed_depth':
                    x = layers.MaxPooling3D(pool_size=2) (x)
                else:
                    x = layers.MaxPooling3D(pool_size=(1, 2, 2)) (x)
            else:
                x = layers.MaxPooling2D(pool_size=2) (x)
    # flatten layer
    x = layers.Flatten() (x)
    # dense layers
    for lyr in range(len(dense_layers)):
        x = layers.Dense(dense_layers[lyr], activation='relu') (x)
        if batch_normalization != False:
            x = layers.BatchNormalization() (x)
        if drop_out != False:
            x = layers.Dropout(drop_out) (x)
    # output layer
    outputs = layers.Dense(num_labels, activation='softmax') (x)
    # declare model
    model = models.Model(inputs, outputs, name='VGG-'+str(sum(conv_layers)+len(dense_layers)+1))
    return model

def model_VGG(inputs, outputs, filter_number=16, drop_out=0.25, initial_learning_rate=1e-5,
              batch_normalization=True, conv_layers=[], dense_layers=[], rnn_layers=[],
              model_type='VGG16', temporal_type='spatio-temporal'):
    '''This function creates VGG-based 2D/3D models
    conv_layers = number of convolution layers for each block
    dense_layers = number of nodes in each dense layers
    model_type = 'VGG16', 'VGG19', 'ECM-paper', 'custom'
    temporal_type = 'spatio_temporal', 'temporal', 'temporal_fixed_depth'
    '''
    input_shape = inputs.shape[1:]
    num_labels = np.max(outputs)+1
    ndim = inputs.ndim-2
    if ndim==2:
        data_type = '2D'
    else:
        data_type = '3D'
    if model_type=='VGG16':
        conv_layers = [2,2,3,3,3]
        dense_layers = [256, 128]
    elif model_type=='VGG19':
        conv_layers = [2,2,4,4,4]
        dense_layers = [256, 128]
    elif model_type=='ECM-paper':
        conv_layers = [2,2,2,2,1]
        dense_layers = []
        initial_learning_rate, filter_number, drop_out, batch_normalization = 1e-5, 16, False, False
        temporal_type = 'spatio_temporal'
    elif model_type=='custom':
        if len(conv_layers)==0 or len(dense_layers)==0:
            raise Exception('Convolution and dense layers must be declared for "Custom" model')
    # model parameters
    parameters = [str(initial_learning_rate), str(batch_normalization), str(drop_out),
                  str(conv_layers), str(rnn_layers), str(dense_layers)]    
    # build model
    tf.keras.backend.clear_session()
    model = VGG_3D(input_shape=input_shape, filter_number=filter_number, num_labels=num_labels, initial_learning_rate = initial_learning_rate,
                   drop_out=drop_out, batch_normalization=batch_normalization,
                   conv_layers=conv_layers, dense_layers=dense_layers, data_type=data_type, temporal_type=temporal_type)
    return model, parameters

#%% RNN Models
def RNN(input_shape=(256, 352, 1), num_labels=2, initial_learning_rate = 1e-5,
        drop_out=False, batch_normalization=False,
        rnn_layers=[128, 256, 512, 128], dense_layers=[32]):
    '''This function builds a RNN given number of rnn and dense layers'''
    # input layer
    inputs = layers.Input(shape=input_shape, dtype=tf.float32, name='video_input')
    # recurrent layers
    for lyr in range(len(rnn_layers)):
        if lyr==0:
            x = layers.CuDNNLSTM(rnn_layers[lyr], return_sequences=True, kernel_initializer='he_normal') (inputs)
        elif lyr==len(rnn_layers)-1:
            x = layers.CuDNNLSTM(rnn_layers[lyr], return_sequences=False, kernel_initializer='he_normal') (x)
        else:
            x = layers.CuDNNLSTM(rnn_layers[lyr], return_sequences=True, kernel_initializer='he_normal') (x)
        if batch_normalization != False:
            x = layers.BatchNormalization() (x)
        if drop_out != False:
            x = layers.Dropout(drop_out) (x)
    # dense layers
    for lyr in range(len(dense_layers)):
        x = layers.Dense(dense_layers[lyr], activation='relu') (x)
        if batch_normalization != False:
            x = layers.BatchNormalization() (x)
        if drop_out != False:
            x = layers.Dropout(drop_out) (x)
    # output layer
    outputs = layers.Dense(num_labels, activation='softmax') (x)
    # declare model
    model = models.Model(inputs, outputs, name='RNN-'+str(len(rnn_layers)+len(dense_layers)))
    return model

def CNN_RNN(input_shape=(16, 256, 352, 1), filter_number=16, num_labels=2, initial_learning_rate = 1e-5,
            conv_layers=[2,2,3,3,3], rnn_layers=[1024,512], dense_layers=[256,128],
            drop_out=False, batch_normalization=False):
    '''This function builds CNN-RNN model given for videos given convolution & dense layers'''
    # input layer
    inputs = layers.Input(shape=input_shape, dtype=tf.float32, name='video_input')
    # recurrent layers
    for lyr in range(len(conv_layers)):
        for conv in range(conv_layers[lyr]):
            if lyr==0:
                x = layers.TimeDistributed(layers.Conv2D((lyr+1)*filter_number, 3, activation='relu', padding='same')) (inputs)
            else:
                x = layers.TimeDistributed(layers.Conv2D((lyr+1)*filter_number, 3, activation='relu', padding='same')) (x)
            if batch_normalization != False:
                x = layers.BatchNormalization() (x)
            if drop_out != False:
                x = layers.Dropout(drop_out) (x)
        # pooling layer
        if lyr != len(conv_layers)-1:
            x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=2)) (x)
    # flatten layer
    x = layers.TimeDistributed(layers.Flatten()) (x)
    # rnn layers
    for lyr in range(len(rnn_layers)):
        if lyr==len(rnn_layers)-1:
            x = layers.GRU(rnn_layers[lyr], return_sequences=False, kernel_initializer='he_normal') (x)
        else:
            x = layers.GRU(rnn_layers[lyr], return_sequences=True, kernel_initializer='he_normal') (x)
        if batch_normalization != False:
            x = layers.BatchNormalization() (x)
        if drop_out != False:
            x = layers.Dropout(drop_out) (x)
    # dense layers
    for lyr in range(len(dense_layers)):
        x = layers.Dense(dense_layers[lyr], activation='relu') (x)
        if batch_normalization != False:
            x = layers.BatchNormalization() (x)
        if drop_out != False:
            x = layers.Dropout(drop_out) (x)
    # output layer
    outputs = layers.Dense(num_labels, activation='softmax') (x)
    # build model
    tf.keras.backend.clear_session()
    model = models.Model(inputs, outputs, name='CNN-RNN-'+str(sum(conv_layers)+len(rnn_layers)+len(dense_layers)))
    return model

def model_RNN(inputs, outputs, filter_number=16, drop_out=0.25, initial_learning_rate=1e-5,
              batch_normalization=True, conv_layers=[], rnn_layers=[], dense_layers=[], 
              architecture_type='RNN', model_type='ECM_paper'):
    '''This function creates RNN/CNN-RNN models
    conv_layers = number of convolution layers for each block
    rnn_layers = number of recurrent layers
    dense_layers = number of nodes in each dense layers
    model_type = 'ECM-paper', 'custom'
    '''
    input_shape = inputs.shape[1:]
    num_labels = np.max(outputs)+1
    if model_type=='ECM-paper':
        rnn_layers = [128, 256, 512, 128]
        dense_layers = [32]
        architecture_type = 'RNN'
        drop_out = 0.2
        batch_normalization = False
    elif model_type=='custom':
        if len(conv_layers)==0 or len(dense_layers)==0 or len(rnn_layers)==0:
            raise Exception('Convolution, recurrent and dense layers must be declared for "Custom" model')
        initial_learning_rate, drop_out, batch_normalization = 1e-5, False, False
    # model parameters
    parameters = [str(initial_learning_rate), str(batch_normalization), str(drop_out),
                  str(conv_layers), str(rnn_layers), str(dense_layers)]    
    tf.keras.backend.clear_session()
    if architecture_type=='RNN':
        model = RNN(input_shape=input_shape, num_labels=num_labels, initial_learning_rate=initial_learning_rate,
                drop_out=drop_out, batch_normalization=batch_normalization, rnn_layers=rnn_layers, dense_layers=[32])
    else:
        model = CNN_RNN(input_shape=input_shape, filter_number=filter_number, num_labels=num_labels,
                        initial_learning_rate=initial_learning_rate, conv_layers=conv_layers, 
                        rnn_layers=rnn_layers, dense_layers=dense_layers,
                        drop_out=drop_out, batch_normalization=batch_normalization)
    return model, parameters

#%% load model with data reshape if necessary
def load_model(inputs, outputs, model_architecture='CNN', model_type='ECM_paper', temporal_type='spatio-temporal',
               initial_learning_rate=1e-5, filter_number=16,  drop_out=0.25, batch_normalization=True,
               conv_layers=[3,3,3], rnn_layers=[128,128], dense_layers=[32], show_model=False):
    '''This model reshape dataset and build model'''
    # check for CNN (VGG-based) model
    if model_architecture=='CNN':
        # input reshape
        inputs = np.expand_dims(inputs, axis=-1)
        # build model
        model, parameters = model_VGG(inputs=inputs, outputs=outputs, filter_number=filter_number,drop_out=drop_out,
                                      initial_learning_rate=initial_learning_rate, batch_normalization=batch_normalization,
                                      conv_layers=conv_layers, dense_layers=dense_layers,
                                      model_type=model_type, temporal_type=temporal_type)
    # check for RNN/CNN-RNN model
    else:
        # data cardinality check
        ndim = inputs.ndim-1
        if ndim != 3:
            raise Exception('Data type is not 3D, please change accordingly...')
        # check for RNN model
        if model_architecture=='RNN' or model_type=='ECM-paper':
            # input reshape
            n, d, h, w = inputs.shape
            inputs = np.reshape(inputs, (n, d*h, w)).astype(np.float32)
        # check for CNN-RNN model
        else:
            # input reshape
            inputs = np.expand_dims(inputs, axis=-1)
        # build model
        model, parameters = model_RNN(inputs=inputs, outputs=outputs, filter_number=filter_number, drop_out=drop_out,
                                      initial_learning_rate=initial_learning_rate, batch_normalization=True, 
                                      conv_layers=conv_layers, rnn_layers=rnn_layers, dense_layers=dense_layers, 
                                      architecture_type=model_architecture, model_type=model_type)
    
    if show_model==True:
        model.summary()
    print('\n-----------------------------------')
    print('Model =', model_type, '['+model_architecture+']')
    print(f'Number of Data = {len(inputs):d} \nData shape = {inputs.shape[1:]}')
    print('-----------------------------------\n')
    return inputs, outputs, model, parameters