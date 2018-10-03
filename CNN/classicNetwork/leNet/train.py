#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:31:43 2018

@author: sadik
"""

import sys
from optparse import OptionParser

sys.path.append('./')

import utils
from utils.process_config import process_config

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure")

(options, args) =  parser.parse_args()

if options.configure:
    conf_file = str(options.configure)
else:
    print('please specify --configure filename')
    exit(0)
    
common_params, dataset_params, net_params, solver_params = process_config(conf_file)
dataset = eval(dataset_params['name'])(common_params, dataset_params)
net = eval(net_params['name'])(common_params, dataset_params)
solver = eval(solver_params['name'])(common_params, dataset_params)
solver.solve()
