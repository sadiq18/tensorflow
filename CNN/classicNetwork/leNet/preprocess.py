#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

#format array to image format 
image_size = 28
num_labels = 10
num_channels = 1 # grayscale
train_path ="../input/train.csv"
test_path ="../input/test.csv"
label_name  = "label"
process_train_path ="../input/train"
process_test_path ="../input/test"
process_valid_path ="../input/valid"
process_sub_path ="../input/sub"

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def main():
    data   = pd.read_csv(train_path)

    dataset = data.drop(label_name, 1)

    labels =data.ix[:,label_name]
    X_train, X_validation,y_train, y_validation = train_test_split(dataset.as_matrix(),
                                                   labels,
                                                   test_size=0.2,
                                                   random_state=0)

    #load submission_data
    submission_dataset = pd.read_csv(test_path)
    train_dataset, train_labels = reformat(X_train, y_train)
    valid_dataset, valid_labels = reformat(X_validation, y_validation)
    submission_dataset = submission_dataset.as_matrix().reshape((-1, image_size, image_size, num_channels)).astype(np.float32)

    print ('Training set   :', train_dataset.shape, train_labels.shape)
    print ('Validation set :', valid_dataset.shape, valid_labels.shape)
    print ('Submission data:', submission_dataset.shape)
    
    with open(process_train_path, 'wb') as f:
        pickle.dump(train_dataset,f)
    with open(process_train_path+"_labels", 'wb') as f:
        pickle.dump(train_labels,f)
        
    with open(process_valid_path, 'wb') as f:
        pickle.dump(valid_dataset,f)
    with open(process_valid_path+"_labels", 'wb') as f:
        pickle.dump(valid_labels,f)

    with open(process_sub_path, 'wb') as f:
        pickle.dump(submission_dataset,f)
        
if __name__ == '__main__':
    main()
    
    

