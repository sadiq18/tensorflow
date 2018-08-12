#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:05:28 2018

@author: sadik
"""

# namespace help in debugging on Tensorboard
import tensorflow as tf

x = tf.placeholder(tf.float32,name='x')

W = tf.Variable([2.5,4.0],tf.float32,name='W')
b = tf.Variable([5.0,10.0],tf.float32,name='b')

with tf.name_scope("Equation_1"):
    y = W * x + b
    
with tf.name_scope("Equation_2"):
    s = W * x
    
with tf.name_scope("mean_equation"):
    final_mean=tf.reduce_mean([y,s],name='final_mean')

# initialize all variables defined
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print "Result (Wx+b) : " , sess.run(y,feed_dict={x:[10,100]})
    print "Result (Wx) : ", sess.run(s,feed_dict={x:[10,100]});
    print "Fianl Mean y,s : ", sess.run(final_mean,feed_dict={x:[10,100]});
    writer = tf.summary.FileWriter("./namespace_demo",sess.graph)
    writer.close()

#execute below command from terminal
# tensorboard --logdir="./namespace_demo"