# encoding: utf-8
# ******************************************************
# Author       : donghoon oh
# Last modified: 2018-08-24
# Filename     : rnn.py
# Description  : RNN Cell with softmax for Spkr Classification
# ******************************************************

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

def build_BRNN(args, inputs, cell_fn, seq_len):
    
    with tf.variable_scope('BRNN'):


        fw_stack_cell = tf.contrib.rnn.MultiRNNCell(
                        [cell_fn(args.num_hidden) for i in range(args.num_layer)])
        
        bw_stack_cell = tf.contrib.rnn.MultiRNNCell(
                        [cell_fn(args.num_hidden) for i in range(args.num_layer)])

        _initial_state_fw = fw_stack_cell.zero_state(args.batch_size, tf.float32)
        _initial_state_bw = bw_stack_cell.zero_state(args.batch_size, tf.float32)
        
        # tensor = [batch_size, time_step, input_feature]
        outputs, output_states =\
                        tf.nn.bidirectional_dynamic_rnn(fw_stack_cell, bw_stack_cell,
                                                        inputs=inputs,
                                                        sequence_length=seq_len,
                                                        initial_state_fw = _initial_state_fw,
                                                        initial_state_bw = _initial_state_bw)
        # rnn outputs 
        # catch hidden of last timestep
        # transpose to [max_time, batch_size, hidden_size]
        output_fw = tf.transpose(outputs[0], [1,0,2])
        output_bw = tf.transpose(outputs[1], [1,0,2])
        # outputs = [batch_size, hidden_size*2] 
        outputs = tf.concat([output_fw[-1], output_bw[-1]], axis=1)
        
        '''
        # use output of all time step
        # outputs = [batch, max_time, hidden_size]
        outputs = tf.concat([outputs[0], outputs[1]], axis=2)
        outputs = tf.reshape(outputs, [args.batch_size, -1]) 
        ''' 
    return outputs 

def build_RNN(args, inputs, cell_fn, seq_len):
    
    with tf.variable_scope('RNN'):


        fw_stack_cell = tf.contrib.rnn.MultiRNNCell(
                        [cell_fn(args.num_hidden) for i in range(args.num_layer)])
        
        _initial_state_fw = fw_stack_cell.zero_state(args.batch_size, tf.float32)
        
        # tensor = [batch_size, time_step, input_feature]
        outputs, output_states =\
                        tf.nn.dynamic_rnn(fw_stack_cell,
                                            inputs=inputs,
                                            sequence_length=seq_len,
                                            initial_state = _initial_state_fw)
        # rnn outputs 
        # catch hidden of last timestep
        # transpose to [max_time, batch_size, hidden_size]
        outputs = tf.transpose(outputs, [1,0,2])[-1]
 
    return outputs 

class BiRNN(object):
    def __init__(self, args, num_steps):
        self.args = args
        self.num_steps = num_steps
        if args.rnncell == 'rnn':
            self.cell_fn = tf.contrib.rnn.BasicRNNCell
        elif args.rnncell == 'gru':
            self.cell_fn = tf.contrib.rnn.GRUCell
        elif args.rnncell == 'lstm':
            self.cell_fn = tf.contrib.rnn.BasicLSTMCell
        else:
            raise Exception("rnncell type not supported: {}".format(args.rnncell))

        self.build_graph(args, num_steps)

    def build_graph(self, args, num_steps):
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            # input = [batch, num_steps, num_feature]
            self.inputs = tf.placeholder(tf.float32,
                            shape=(None, num_steps, args.num_feature))
            
            # target = [batch, dim_targets]
            self.targets = tf.placeholder(tf.int32, shape=(None))
            targets_onehot = tf.one_hot(self.targets, args.num_class)
            self.seq_len = tf.placeholder(tf.int32, shape=(None))
            
            self.config = {'name': args.model,
                           'rnncell': self.cell_fn,
                           'num_layer': args.num_layer,
                           'num_hidden': args.num_hidden,
                           'num_class': args.num_class,
                           'learning rate': args.learning_rate,
                           'keep prob': args.keep_prob,
                           'batch size': args.batch_size}

            outputs = build_BRNN(self.args, self.inputs, self.cell_fn, self.seq_len) 
            #outputs = build_RNN(self.args, self.inputs, self.cell_fn, self.seq_len) 
            
            # fc layer
            logits = tf.contrib.layers.fully_connected(outputs, args.num_class,
                                    activation_fn=tf.nn.relu)
            logits = tf.contrib.layers.fully_connected(logits, args.num_class,
                                    activation_fn=None)
            
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits, labels=targets_onehot))
            
            self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
            
            # parameters of model 
            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()
                 
            # prediction
            self.prediction = tf.argmax(logits,1)
            self.truth = tf.argmax(targets_onehot,1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.truth), tf.float32))

            # initialization 
            self.initial_op = tf.global_variables_initializer()
            # save  
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)
