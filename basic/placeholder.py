#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 16:29:37 2018

@author: sadik
"""

import tensorflow as tf

# placeholder
# placeholder can be assigned later before running function
x = tf.placeholder(tf.int32,shape=[3],name='x')
y = tf.placeholder(tf.int32,shape=[3],name='y')


#space is not allowed for name
# element wise sum of x vector
sum_x = tf.reduce_sum(x, name='sum_x')


#element wise multiply of y vector
prod_y = tf.reduce_prod(y, name='prod_y')

# sum_x/prod_y
final_div = tf.div(sum_x,prod_y, name='final_div')

# (sum_x + prod_y)/2
final_mean = tf.reduce_mean([sum_x,prod_y],name='final_mean')

sess = tf.Session()

# printing tensor
print 'X:' ,sess.run(x,feed_dict={x:[100,200,300]})
print 'Y:' ,sess.run(y,feed_dict={y:[1,2,3]})
print 'sum_x:' ,sess.run(sum_x,feed_dict={x:[100,200,300]})
print 'prod_y:' ,sess.run(prod_y,feed_dict={y:[1,2,3]})
print 'final_mean:' ,sess.run(final_mean,feed_dict={x:[100,200,300],y:[1,2,3]})

# close session
sess.close()