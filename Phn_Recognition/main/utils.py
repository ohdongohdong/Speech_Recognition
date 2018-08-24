# encoding: utf-8
# ******************************************************
# Author       : donghoon oh
# Last modified: 2018-08-20
# Filename     : utils.py
# Description  : Function utils library for Speech Recognition
# ******************************************************

import time
import os
import numpy as np
import tensorflow as tf
import math
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def check_path_exists(path):
    """ check a path exists or not
    """
    if isinstance(path, list):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)
    else:
        if not os.path.exists(path):
            os.makedirs(path)

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth =True
    return config

def load_data(path):
    # read data
    print('read data')
    input_list = []
    label_list = []
    for path, subdir, files in os.walk(path):
        phn_class =  path.split('/')[-1]
        if phn_class =='train' or phn_class == 'test':
            continue 
        print('phn class : {}'.format(path.split('/')[-1]))
        for f in files:
            file_name = os.path.join(path, f)
            if f.endswith('.npy'):
                input_list.append(np.load(file_name))
                label_list.append(phn_class)
    
    assert len(input_list) == len(label_list)

    # dimensions of inputList:batch*39*time_length
    nFeatures = input_list[0].shape[0]
    print(nFeatures)

    # find the max time_length
    maxLength = 0
    seq_len_list = []
    for inp in input_list:
        seq_len_list.append(inp.shape[1])
        maxLength = max(maxLength, inp.shape[1])
    print(maxLength)
    # padding
    print('padding')
    for i, inp in enumerate(input_list):
        if i % 1000 == 0:
            print('{}/{}'.format(i, len(input_list)))
        # padSecs is the length of padding
        padSecs = maxLength - inp.shape[1]
	# numpy.pad pad the inputList[origI] with zeros at the tail
        input_list[i] = np.pad(inp.T, ((0,padSecs), (0,0)), 'constant', constant_values=0)
        
    pad_input_list = np.asarray(input_list)
    print('input shape : {}'.format(pad_input_list.shape))
        
    # phn label to idx
    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw', \
                 'v', 'w', 'y', 'z', 'zh'] 
    label_index_list = [group_phn.index(label) for label in label_list]
    label_index_list = np.array(label_index_list)
    print('label shape : {}'.format(label_index_list.shape)) 
    return pad_input_list, label_index_list, seq_len_list

def count_params(model, mode='trainable'):
    ''' count all parameters of a tensorflow graph
    '''
    if mode == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_op])
    elif mode == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_trainable_op])
    else:
        raise TypeError('mode should be all or trainable.')
    print('number of '+mode+' parameters: '+str(num))
    return num

def next_batch(batch_size, data_set):
    '''
    split all of data to batch set
    '''  
    idx = np.arange(0, len(data_set[0]))
    np.random.shuffle(idx)
    idx = idx[:batch_size]

    batch_data_set = []
    for data in data_set:
        batch_data_set.append(np.asarray([data[i] for i in idx]))

    return batch_data_set

def logging(model, logfile, epoch=0, num_epoch=100, batch=0, num_batch=256, 
                    loss=0.0, accuracy=0.0, delta_time=0, mode='train'):
    ''' log the cost and error rate and time while training or testing
    '''  
    if mode == 'config':
        with open(logfile, "a") as myfile:
            myfile.write('\n'+str(time.strftime('%X %x %Z'))+'\n')
            myfile.write('\n'+str(model.config)+'\n')

    elif mode == 'epoch':
        with open(logfile, "a") as myfile:
            myfile.write('\n[Epoch :{}]\n'.format(epoch+1))
    
    elif mode == 'batch':
        with open(logfile, "a") as myfile:
            myfile.write('batch: {}/{}, loss={:.4f}, acc={:.3f}\n'.format(batch+1, num_batch, loss, accuracy))
    
    elif mode =='train': 
        with open(logfile, "a") as myfile:
            myfile.write('==> Epoch: {}/{}, loss={:.4f}, acc={:.3f}, epoch time : {}\n'\
                                .format(epoch+1, num_epoch, loss, accuracy, delta_time)) 
    else:
        raise TypeError('mode should be write right.')

def confusion_matrix(truth, prediction, save_path):
    phn_class = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw', \
                 'v', 'w', 'y', 'z', 'zh'] 
    
    cm = confusion_matrix(truth, prediction)
    
    # normalization
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('shape of confusion matrix : {}'.format(cm.shape))
    
    # figure
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix of phn recogniton')
    plt.colorbar()
    tick_marks = np.arange(len(phn_class))
    plt.xticks(tick_marks, phn_class)
    plt.yticks(tick_marks, phn_class)

    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], '.2f'),
                    horizontalalignment='center',
                    color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('Truth')
    plt.xlabel('Prediction')
    plt.savefig(save_path)

    return cm


