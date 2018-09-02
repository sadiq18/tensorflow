#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 13:48:05 2018

@author: sadik
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

classifier=Sequential()

classifier.add(Conv2D(32,(3,3),strides=2,
                 input_shape=(50,50,3),
                 activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))


classifier.add(Conv2D(64,(3,3),strides=2,
                 activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(256,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

classifier.summary()

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=.2,
                                   zoom_range=.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/home/sadik/dataset/train',
                                                 target_size=(50,50),
                                                 batch_size=32,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('/home/sadik/dataset/test',
                                                 target_size=(50,50),
                                                 batch_size=32,
                                                 class_mode='binary')


classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         validation_data=test_set,
                         validation_steps=2000)

 
 





