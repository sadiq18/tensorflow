#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 18:28:05 2018

@author: sadik
"""

"""
    Solver Abstract class
"""

class Solver(object):
    
    def __init__(self, dataset, net, common_params, solver_params):
        """
            common_params: A params dict
            solver_params: A params dict
        """
        raise NotImplementedError
        
    def solve(self):
        raise NotImplementedError