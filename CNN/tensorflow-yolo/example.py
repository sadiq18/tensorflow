#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:45:52 2018

@author: sadik
"""

import sys

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet
import tensorflow as tf
import cv2
import numpy as np

classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa","train","tvmonitor"]

def process_predicts(predicts):
    p_classes = predicts[0, :, :, 0:20]
    C = predicts[0, :, :, 20:22]
    coordinate = predicts[0, :, :, 22:]
    
    p_classes = np.reshape(p_classes, (7, 7, 1, 20))
    C = np.reshape(C, (7, 7, 1, 20))
    
    P = C * p_classes
    
    index = np.argmax(P)
    
    index = np.unravel_index(index, P.shape)
    
    class_num = index[3]
    
    coordinate = np.reshape(coordinate, (7, 7, 2, 4))
    
    max_coordinate = coordinate[index[0], index[1], index[2], :]
    
    xcenter = max_coordinate[0]
    ycenter = max_coordinate[1]
    
    w = max_coordinate[2]
    h = max_coordinate[3]
    
    xcenter = (index[1] + xcenter) * (448/7.0)
    ycenter = (index[0] + ycenter) * (448/7.0)
    
    w = w * 448
    h = h * 448
    
    xmin = xcenter - w/2.0
    ymin = ycenter - h/2.0
    
    xmax = xmin + w
    ymax = ymin + w
    
    return xmin, ymin, xmax, ymax, class_num

common_params = {'image_size' :  448, 'num_classes' : 20, 'batch_size' : 1}
net_params = {'cell_size' :  7, 'boxes_per_cell' : 2, 'weight_decay' : 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.interference(image)

with tf.Session() as sess:
    saver = tf.train.Saver(net.trainable_collection)
    saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')
    #assigne image array to of shape (1, 448, 448, 3) to np_img
    np_image = None
    np_predicts = sess.run(predicts, feed_dict={image: np_image})
    
    xmin, ymin, xmax, ymax, class_num = process_predicts(np_predicts)
    
    class_name = classes_name[class_num]

