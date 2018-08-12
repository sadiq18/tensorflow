#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 16:40:31 2018

@author: sadik
"""

import tensorflow as tf

# constant => Immutable values which does not change
# placeholder => Assigned once and does not change after
# Variable => Mutable Tensor values that persist across multiple calls to session.run()

# y = Wx + b

x = tf.placeholder(tf.float32,name='x')

W = tf.Variable([2.5,4.0],tf.float32,name='W')
b = tf.Variable([5.0,10.0],tf.float32,name='b')

y = W * x + b

# initialize all variables defined
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print "Result (Wx+b) : " , sess.run(y,feed_dict={x:[10,100]})
    
# s = Wx
s = W * x

# initialize W only
init = tf.variables_initializer([W])

with tf.Session() as sess:
    sess.run(init)
    print "Result (Wx) : ", sess.run(s,feed_dict={x:[100,1000]});
    

number = tf.Variable(2)
multiplier = tf.Variable(1)

init = tf.global_variables_initializer()

# print number * mutiplier
#number = number * multiplier
#mutiplier = multiplier + 1
result = number.assign(tf.multiply(number,multiplier))

with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        print sess.run(result)
        sess.run(multiplier.assign_add(1))