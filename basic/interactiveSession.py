#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:32:27 2018

@author: sadik
"""

import tensorflow as tf

sess = tf.InteractiveSession()

A = tf.constant([4],tf.int32,name='A')
X = tf.placeholder(tf.int32,name='X')

y = A*X

print y.eval(feed_dict={X:[5]})

sess.close()