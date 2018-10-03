#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 18:52:38 2018

@author: sadik
"""

class Dataset(object):
    """
        Base Datasset
    """
    
    def __init__(self, common_params, dataset_params):
        """
            common_params: A params dict
            dataset_params: A params dict
        """
        raise NotImplementedError
        
    def batch(self):
        """
            Get batch
        """
        raise NotImplementedError