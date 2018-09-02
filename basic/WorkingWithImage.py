#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 09:04:13 2018

@author: sadik
"""

import tensorflow as tf
import matplotlib.image as mp_img
import matplotlib.pyplot as plot

filename='img.jpg'

image = mp_img.imread(filename)
print "Image shape: ",image.shape

plot.imshow(image)
plot.show()

X = tf.Variable(image,name='X')
init = tf.global_variables_initializer()

# TO transpose Image
with tf.Session() as sess:
    sess.run(init)
    
    # perm order for image [0,1,2]
    # change order to transpose
    transpose = tf.transpose(X,perm=[1,0,2])
    
    result=sess.run(transpose)
    
    print "Transpose Image Shape: ",result.shape
    plot.imshow(result)
    plot.show()
    
    