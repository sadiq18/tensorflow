#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 16:11:57 2018

@author: sadik
"""

import tensorflow as tf
import numpy as np

zeroD = np.array(30,dtype=np.int32);

sess = tf.Session();

print tf.rank(zeroD);
print tf.shape(zeroD);

tensor = tf.constant(zeroD,name='numpy_tensor')
print tf.rank(tensor);
print tf.shape(tensor);

sess.close()