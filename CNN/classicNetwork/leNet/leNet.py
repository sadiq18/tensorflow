#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 19:22:27 2018

@author: sadik
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

import network
from network.net import Net

class LeNet5(Net):
    def __init__(self, commom_params, net_params, test=False):
        super(LeNet5, self).__init__(commom_params,net_params)
        
        #process params
        self.image_size = int(commom_params['image_size'])
        self.num_classes = int(commom_params['num_class'])
        self.batch_size = int(commom_params['batch_size'])
        self.weight_decay = float(commom_params['weight_decay'])
        self.kernelSize = int(net_params['kernelSize'])
        self.depth1Size = int(net_params['depth1Size'])
        self.depth2Size = int(net_params['depth2Size'])
        self.keep_prob = float(net_params['keep_prob'])
        
    def inference(self, images):
        #C1
        conv_num=1
        h_conv = self.conv2d("conv%d" %(conv_num), images, [self.kernelSize,self.kernelSize,1,self.depth1Size], stride=1)
        #S2
        h_pool = self.max_pool(h_conv,[2,2], stride=2)
        conv_num+=1
        
        #C3
        h_conv =self.conv2d("conv%d" %(conv_num), images, [self.kernelSize,self.kernelSize,1,self.depth2Size], stride=1)
        #S4
        h_pool = self.max_pool(h_conv,[2,2], stride=2)
        
        #h_pool_reshaped = tf.reshape(h_pool,[shape[0],shape[1]*shape[2]*shape[3]])
        shape = h_pool.get_shape().as_list()
        local1 = self.local('local1', h_pool, shape[1]*shape[2]*shape[3], 120)
        local1 = tf.nn.dropout(local1, keep_prob=self.keep_prob)
        
        local2 = self.local('local2', local1, 120, 84)
        local2 = tf.nn.dropout(local2, keep_prob=self.keep_prob)
        
        local3 = self.local('local3', local2, 84, 1)
        local3 = tf.nn.dropout(local3, keep_prob=self.keep_prob)
        
        model=local3
        return model
    
    def loss(self, predicts, labels):
        return (100.0 * np.sum(np.argmax(predicts, 1) == np.argmax(labels, 1))
          / predicts.shape[0])