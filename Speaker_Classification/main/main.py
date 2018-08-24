# encoding: utf-8
# ******************************************************
# Author       : Donghoon oh
# Last modified: 2018-08-24
# Filename     : main.py
# Description  : Training models on TIMIT dataset for Speaker recognition
#
# Data : TIMIT
# Input : phoneme wave file
# Output : speaker
# ******************************************************

import time
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'

import numpy as np
import tensorflow as tf

from utils import *
from model.rnn import BiRNN

from tensorflow.python.platform import flags
from tensorflow.python.platform import app

flags.DEFINE_string('mode', 'train', 'set whether to train or test')
flags.DEFINE_string('model', 'BiRNN', 'set the model to use, BiRNN, CNN')
flags.DEFINE_string('rnncell', 'lstm', 'set the rnncell to use, rnn, gru, lstm...')
flags.DEFINE_integer('num_layer', 3, 'set the layers for rnn')
flags.DEFINE_boolean('layerNormalization', True, 'set whether to apply layer normalization to rnn cell')

flags.DEFINE_integer('batch_size', 64, 'set the batch size')
flags.DEFINE_integer('num_hidden', 512, 'set the hidden size of rnn cell')
flags.DEFINE_integer('num_feature', 39, 'set the size of input feature')
flags.DEFINE_integer('num_class', 462, 'set the speakrs')
flags.DEFINE_integer('num_epoch', 200, 'set the number of epochs')
flags.DEFINE_float('learning_rate', 0.0001, 'set the learning rate')
flags.DEFINE_float('keep_prob', 0.8, 'set probability of dropout')
flags.DEFINE_float('grad_clip', -1, 'set the threshold of gradient clipping, -1 denotes no clipping')
flags.DEFINE_string('datadir', '../data', 'set the data root directory')
flags.DEFINE_string('logdir', '../log', 'set the log directory')

FLAGS = flags.FLAGS

# set arguments
mode = FLAGS.mode
rnncell = FLAGS.rnncell
num_layer = FLAGS.num_layer

batch_size = FLAGS.batch_size
num_hidden = FLAGS.num_hidden
num_feature = FLAGS.num_feature
num_class = FLAGS.num_class
num_epoch = FLAGS.num_epoch
learning_rate = FLAGS.learning_rate
keep_prob = FLAGS.keep_prob
grad_clip = FLAGS.grad_clip
datadir = FLAGS.datadir

# mode check
print('[{} mode]'.format(mode))
if mode == 'test':
    batch_size = 64
    num_epoch = 1
    keep_prob = 1.0
    is_training = False
else:
    is_training = True

# set path of log directory
logdir = os.path.join(FLAGS.logdir, FLAGS.model)
savedir = os.path.join(logdir, 'save')
resultdir = os.path.join(logdir, 'result')
loggingdir = os.path.join(logdir, 'logging')
check_path_exists([logdir, savedir, resultdir, loggingdir])

# set directory of data
data_dir = os.path.join(datadir, mode, 'mfcc')

checkpoint_path = os.path.join(savedir, 'model.ckpt')
logfile = os.path.join(loggingdir, str(datetime.datetime.strftime(datetime.datetime.now(),
    '%Y-%m-%d_%H:%M:%S') + '_' + mode + '.txt').replace(' ', '').replace('/', ''))

# Run
class Runner(object):
    # set configs
    def _default_configs(self):
        return {'mode': mode,
                'rnncell': rnncell,
                'batch_size': batch_size,
                'num_hidden': num_hidden,
                'num_feature': num_feature,
                'num_class': num_class,
                'num_layer': num_layer,
                'num_epoch' : num_epoch, 
                'learning_rate': learning_rate,
                'keep_prob': keep_prob,
                'grad_clip': grad_clip,
                'is_training':is_training
                }

    # train
    def train(self, args, model, input_train, target_train, seq_train):
        
        with tf.Session(graph=model.graph, config=get_tf_config()) as sess:
            # initialization 
            sess.run(model.initial_op)
            
            for epoch in range(num_epoch):
                start = time.time()
                print('\n[Epoch :{}]\n'.format(epoch+1))
                logging(model=model, logfile=logfile, epoch=epoch, mode='epoch')
                
                # mini batch 
                batch_epoch = int(input_train.shape[0]/batch_size)
                
                batch_loss = np.zeros(batch_epoch)
                batch_acc = np.zeros(batch_epoch)
                for b in range(batch_epoch):
                    
                    batch_inputs, batch_targets, batch_seq_len = next_batch(
                                    batch_size, [input_train, target_train, seq_train])  
                    ''' 
                    batch_inputs = input_train[:batch_size]                    
                    batch_targets = target_train[:batch_size]                    
                    batch_seq_len = seq_train[:batch_size]                    
                    ''' 
                    feed = {model.inputs:batch_inputs,
                            model.targets:batch_targets,
                            model.seq_len:batch_seq_len}

                    _, l, acc = sess.run([model.optimizer, model.loss, model.accuracy],
                                        feed_dict=feed)

                    batch_loss[b] = l
                    batch_acc[b] = acc
                    if b%10 == 0:
                        print('batch: {}/{}, loss={:.3f}, accuracy={:.3f}'.format(b+1, batch_epoch, l, acc))
                        logging(model, logfile, batch=b, num_batch=batch_epoch, loss=l, accuracy=acc, mode='batch')
                
                loss = np.sum(batch_loss)/batch_epoch
                accuracy = np.sum(batch_acc)/batch_epoch

                delta_time = time.time()-start
                print('\n==> Epoch: {}/{}, loss={:.4f}, accuracy={:.3f}, epoch time : {}\n'\
                                .format(epoch+1, num_epoch, loss, accuracy, delta_time))
                logging(model, logfile, epoch, num_epoch,
                            loss=loss, accuracy=accuracy, delta_time=delta_time, mode='train')
                
                # save model by epoch
                model.saver.save(sess, checkpoint_path) 

    def test(self, args, model, input_test, target_test, seq_test):

        with tf.Session(graph=model.graph, config=get_tf_config()) as sess: 
            # initialization 
            sess.run(model.initial_op)
            epoch = 1
            
            # load check point
            model.saver.restore(sess, checkpoint_path)
         
            for epoch in range(num_epoch):
                start = time.time()
                print('\n[Epoch :{}]\n'.format(epoch+1))
                logging(model=model, logfile=logfile, epoch=epoch, mode='epoch')
                
                # mini batch 
                batch_epoch = int(input_test.shape[0]/batch_size)
                
                batch_loss = np.zeros(batch_epoch)
                batch_acc = np.zeros(batch_epoch)
                total_pred = []
                total_truth = []
                for b in range(batch_epoch):
                    
                    batch_inputs, batch_targets, batch_seq_len = next_batch(
                                    batch_size, [input_test, target_test, seq_test])  
                    
                    feed = {model.inputs:batch_inputs,
                            model.targets:batch_targets,
                            model.seq_len:batch_seq_len}

                    l, acc, p, t = sess.run([model.loss,model.accuracy,
                                        model.prediction, model.truth],
                                        feed_dict=feed)
                    batch_loss[b] = l
                    batch_acc[b] = acc
                    total_pred.extend(p)
                    total_truth.extend(t)
                    if b%10 == 0:
                        print('batch: {}/{}, loss={:.4f}, accuracy={:.4f}'.format(b+1, batch_epoch, l,acc))
                        logging(model, logfile, batch=b, num_batch=batch_epoch, loss=l, accuracy=acc, mode='batch')
                    
                loss = np.sum(batch_loss)/batch_epoch 
                accuracy = np.sum(batch_acc)/batch_epoch 
                delta_time = time.time()-start
                print('\n==> Epoch: {}/{}, loss={:.4f}, accuracy={:.4f}, epoch time : {}\n'\
                                .format(epoch+1, num_epoch, loss, accuracy, delta_time))
                logging(model, logfile, epoch, num_epoch,
                            loss=loss, accuracy=accuracy, delta_time=delta_time, mode='train')
                
    
    # main
    def run(self):
        # set args
        args_dict = self._default_configs()
        args = dotdict(args_dict)
         
        # step 1 
        # load preprocessed data
        input_data, label_data, seq_len_data = load_data(data_dir)
       
        # input = [batch, num_feature, num_steps]
        print('[model data set]') 
        print('shape of input : {}'.format(input_data.shape))
        print('shape of target : {}'.format(label_data.shape))
        
        # data parameters
        num_steps = input_data.shape[1]

        # load  model 
        model = BiRNN(args, num_steps)

        # count the num of parameters
        num_params = count_params(model, mode='trainable')
        all_num_params = count_params(model, mode='all')
        model.config['trainable params'] = num_params
        model.config['all params'] = all_num_params
        print('\n[model information]\n')
        print(model.config)  
        
        # [step 3]
        # learning 
        logging(model=model, logfile=logfile, mode='config')
        
        if mode == 'train':
            self.train(args, model, input_data, label_data, seq_len_data)
        elif mode == 'test':
            self.test(args, model, input_data, label_data, seq_len_data)

if __name__ == '__main__':
    runner = Runner()
    runner.run()
