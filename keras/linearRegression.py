#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 09:45:08 2018

@author: sadik
"""

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

train_X = np.linspace(-1,1,1001)
train_y = 4 * train_X + np.random.randn(1001)*.44

# Intantiate a model
model = Sequential()

# Add Layers
model.add(Dense(input_dim=1,
                output_dim=1,
                init='uniform',
                activation='linear'))

print model.summary()
print "Weights and bias before train"
weights = model.layers[0].get_weights()
print "W : ", weights[0]
print "b : ", weights[1]

# Compile Model
model.compile(optimizer='sgd',loss='mse')

# Train Model
model.fit(train_X,train_y,epochs=50,verbose=1)

print "Weights and bias after train"
weights = model.layers[0].get_weights()
print "W : ", weights[0]
print "b : ", weights[1]


# Use Model
result =  model.predict(train_X)

plt.scatter(train_X,train_y,label='Data',color=['red'])
plt.plot(train_X,result,label='Prediction')
plt.show()