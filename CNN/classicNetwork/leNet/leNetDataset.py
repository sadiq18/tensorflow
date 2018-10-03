#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 20:44:57 2018

@author: sadik
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np

from network.dataset import Dataset

class ImageDataset(Dataset):
    
    def __init__(self, common_params, dataset_params):
        """
            Args:
                common_params: A dict
                dataset_params: A dict
        """
        #process params
        self.data_path = str(dataset_params['path'])
        self.width = int(common_params['image_size'])
        self.height = int(common_params['image_size'])
        self.batch_size = int(common_params['batch_size'])
        
   
    def batch(self):
        """get batch
            Returns:
                images: 4-D ndarray [batch_size, height, width, 3]
                labels: 3-D ndarray [batch_size, max_objects, 5]
        """
        
        with open(self.data_path,'rb') as f:
            images = pickle.load(f)
            
        with open(self.data_path+"_labels",'rb') as f:
            labels = pickle.load(f)
            
        start=0
        
        while True:
            if start+self.batch_size <= len(images):
                if start == 0:
                    yield images, labels
                else:
                    temp = start
                    start=0
                    yield images[temp:], labels[temp:]
            temp = start
            start += self.batch_size
            yield images[temp:temp+self.batch_size], labels[temp:temp+self.batch_size]
    