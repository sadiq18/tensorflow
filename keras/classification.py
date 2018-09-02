#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 11:31:18 2018

@author: sadik
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import np_utils

iris = sns.load_dataset("iris")
iris = pd.concat([iris,pd.get_dummies(iris['species'], prefix='species')],axis=1)
iris.drop(['species'],axis=1, inplace=True)

X = iris.values[:,:4]
y = iris.values[:,4:]

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=.3,random_state=42)

model = Sequential()
model.add(Dense(16,input_shape=(4,),
                name='Input_Layer',
                activation='relu'))

model.add(Dense(3,name='Output_Layer',
                activation='softmax'))

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_X,train_y,epochs=20,batch_size=2,verbose=1)

result = model.predict(test_X)
