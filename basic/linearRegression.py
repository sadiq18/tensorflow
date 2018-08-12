#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:42:15 2018

@author: sadik
"""

import tensorflow as tf

#Model Parameter
W = tf.Variable([.3],dtype=tf.float32)
b = tf.Variable([-.3],dtype=tf.float32)

#Model input and output
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#linear model
linear_model = W*X + b

#loss
loss = tf.reduce_mean(tf.square(linear_model-y))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

#training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(1000):
        sess.run(train,feed_dict={X:x_train,y:y_train})
        # evaluate accuracy
        curr_w,curr_b,curr_loss=sess.run([W,b,loss],feed_dict={X:x_train,y:y_train})
        print "w:%s b:%s loss:%s" %(curr_w,curr_b,curr_loss)