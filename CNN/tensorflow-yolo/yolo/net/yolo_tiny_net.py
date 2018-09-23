#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 17:28:51 2018

@author: sadik
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from yolo_net import YoloNet

class YoloTinyNet(YoloNet):
    def __init__(self, commom_params, net_params, test=False):
        """
            common params: a params dict
            net_params: a params dict
        """
        super(YoloTinyNet, self).__init__(commom_params,net_params, test)
        
    def inference(self, images):
        """Build YOLO model
            
            Args:
                images: 4-D tensor [batch_size, image_height, image_width, channels]
                
            Return:
                predicts:4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        """
            
        conv_num = 1
        temp_conv = self.conv2d("conv%d" %(conv_num), images, [3,3,3,16], stride=2)
        conv_num +=1
        temp_conv = self.max_pool(temp_conv, [2,2], stride=2)
            
        temp_conv = self.conv2d("conv%d" %(conv_num), temp_conv, [3,3,16,32])
        conv_num +=1
        temp_conv = self.max_pool(temp_conv, [2,2], stride=2)
            
        temp_conv = self.conv2d("conv%d" %(conv_num), temp_conv, [3,3,32,64])
        conv_num +=1
        temp_conv = self.max_pool(temp_conv, [2,2], stride=2)
        
        temp_conv = self.conv2d("conv%d" %(conv_num), temp_conv, [3,3,64,128])
        conv_num +=1
        temp_conv = self.max_pool(temp_conv, [2,2], stride=2)
        
        temp_conv = self.conv2d("conv%d" %(conv_num), temp_conv, [3,3,128,256])
        conv_num +=1
        temp_conv = self.max_pool(temp_conv, [2,2], stride=2)
        
        temp_conv = self.conv2d("conv%d" %(conv_num), temp_conv, [3,3,256,512])
        conv_num +=1
        temp_conv = self.max_pool(temp_conv, [2,2], stride=2)
        
        temp_conv = self.conv2d("conv%d" %(conv_num), temp_conv, [3,3,512,1024])
        conv_num +=1
        temp_conv = self.max_pool(temp_conv, [2,2], stride=2)
        
        for i in range(2):
            temp_conv = self.conv2d("conv%d" %(conv_num), temp_conv, [3,3,1024,1024])
            conv_num +=1
        
        temp_conv = tf.transpose(temp_conv, (0,3,1,2))
            
        #Fully Connected Layer
        local1 = self.local('local1', temp_conv, self.cell_size * self.cell_size * 1024, 256)
        
        local2 = self.local('local2' ,local1, 256, 4096)
        
        local3 = self.local('local3' ,local2, 4096,
                            self.cell_size * self.cell_size * (self.num_classes + 5 * self.boxes_per_cell), leaky=False)
        
        n1 = self.cell_size * self.cell_size * self.num_classes
        n2 = self.cell_size * self.cell_size * self.boxes_per_cell
        
        class_probs = tf.reshape(local3[:, 0:n1], (-1, self.cell_size, self.cell_size, self.num_classes))
        scales = tf.reshape(local3[:, n1:n2], (-1, self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = tf.reshape(local3[:, n2:], (-1, self.cell_size, self.cell_size, self.boxes_per_cell * 4))
        
        local3 = tf.concat([class_probs, scales, boxes], 3)
        predicts = local3
            
        return predicts