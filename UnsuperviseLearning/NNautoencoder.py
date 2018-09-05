#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 10:08:58 2018

@author: sadik
"""
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA

n_inputs=3
n_hiddens=3
n_outputs=n_inputs
learning_rate=0.01
path='/home/sadik/dataset/stocks.csv'

data=pd.read_csv(path)
data['Date']=pd.to_datetime(data['Date'],infer_datetime_format=True)
data=data.sort_values(['Date'],ascending=['True'])
data = data[['AAPL','GOOG','NFLX']]

returns = data[[key for key in dict(data.dtypes) \
                if dict(data.dtypes)[key] in ['float64','int64']]].pct_change()
returns=returns[1:]
returns_arr=returns.as_matrix()[:20]

scalar = StandardScaler()
returns_arr_scaled = scalar.fit_transform(returns_arr)
result=PCA(returns_arr_scaled,standardize=True)
print result.fracs
print result.Y.shape
print result.Wt.shape

tf.reset_default_graph()

x=tf.placeholder(tf.float32,shape=[None,n_inputs])
hidden=tf.layers.dense(x,n_hiddens)
output=tf.layers.dense(hidden,n_outputs)

reconstruction_loss=tf.reduce_mean(tf.square(output-x))

optimizer=tf.train.AdamOptimizer(learning_rate)
training_op=optimizer.minimize(reconstruction_loss)

init=tf.global_variables_initializer()
n_iteration=10000

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iteration):
        training_op.run(feed_dict={x:returns_arr_scaled})
    output_val=output.eval(feed_dict={x:returns_arr_scaled})
    print output_val.shape