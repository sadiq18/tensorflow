#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 18:51:02 2018

@author: sadik
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