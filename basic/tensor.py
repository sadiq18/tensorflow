#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 15:40:21 2018

@author: sadik
"""

import tensorflow as tf

x = tf.constant([100,200,300],name='x')
y = tf.constant([1,2,3],name='y')
print "rank of x:",tf.rank(x)
print "shape of x:",tf.shape(x)
print "rank of y:",tf.rank(y)
print "shape of y:",tf.shape(x)


#space is not allowed for name
# element wise sum of x vector
sum_x = tf.reduce_sum(x, name='sum_x')
print tf.rank(sum_x)

#element wise multiply of y vector
prod_y = tf.reduce_prod(y, name='prod_y')

# sum_x/prod_y
final_div = tf.div(sum_x,prod_y, name='final_div')

# (sum_x + prod_y)/2
final_mean = tf.reduce_mean([sum_x,prod_y],name='final_mean')

sess = tf.Session()

# printing tensor
print 'X:' ,sess.run(x)
print 'Y:' ,sess.run(y)
print 'sum_x:' ,sess.run(sum_x)
print 'prod_y:' ,sess.run(prod_y)
print 'final_div:' ,sess.run(final_div)
print 'final_mean:' ,sess.run(final_mean)

sess.close()

