#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 15:24:32 2018

@author: sadik
"""

import numpy as np
import tensorflow as tf

#import matplotlib.pyplot as plt

input_1d_x = np.array([1,2,3.0,4,5,126,21,33,6,73.0,2,3,56,98,100,4,8,33,102],dtype=np.float32)

def input_fn_1d(input_1d):
    input_t=tf.convert_to_tensor(input_1d,tf.float32)
    input_t=tf.expand_dims(input_t,1)
    return (input_t,None)

#plt.scatter(input_1d_x,np.zeros_like(input_1d_x),s=500)
#plt.show()
k_means_estimator=tf.contrib.factorization.KMeansClustering(num_cluster=2)

fit = k_means_estimator.train(lambda : input_fn_1d(input_1d_x),
                               steps=1000)

cluster_id = k_means_estimator.cluster_centers()
print cluster_id

ex_id_x=np.array([0,100],dtype=np.float32)
cluster_indices = list(k_means_estimator.predict_cluster_index(lambda : input_fn_1d(ex_id_x)))
print cluster_indices



