#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:06:38 2018

@author: sadik
"""

import os
import xml.etree.ElementTree as ET
import struct
import numpy as np

classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa","train","tvmonitor"]

classes_num = {classes_name[i]:i for i in range(len(classes_name))}

YOLO_ROOT = os.path.abspath("./")
DATA_PATH = os.path.join(YOLO_ROOT, 'data/')
OUTPUT_PATH = os.path.join(YOLO_ROOT, 'data/output.txt')

def convert_to_string(image_path, labels):
    """
        convert image path, labels to string
        
        Return:
            string
    """
    
    out_string = ''
    out_string += image_path
    for label in labels:
        for i in labels:
            out_string +=' ' + str(i)
    out_string += '\n'
    return out_string

def main():
    out_file = open(OUTPUT_PATH, 'w')
    
    in_file = DATA_PATH + ""
    
    data_list = os.listdir(in_file)
    
    data_list = [in_file + tmp for tmp in data_list]
    
    for data in data_list:
        try:
            image,labels = data["iamge"],data["labels"]
            record = convert_to_string(image, labels)
            out_file.write(record)
        except:
            pass
        out_file.close()
        
if __name__ == '__main__':
    main()
    
    

