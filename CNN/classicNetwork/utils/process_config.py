#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 18:55:17 2018

@author: sadik
"""

import ConfigParser

def process_config(config_file):
    """
        process configure file to generate commonparams, datasetparams, netparams
        
        Args:
            conf_file: configure file path
        Returns:
            CommonParams, DatasetParams, NetParams, SolverParams
    """
    
    common_params = {}
    dataset_params = {}
    net_params = {}
    solver_params = {}
    
    #configure_parser
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    
    #sections and options
    for section in config.sections():
        #construct common parser
        if section == 'Common':
            for option in config.options(section):
                common_params[option] = config.get(section, option)
        #construct dataset params
        elif section == 'Dataset':
            for option in config.options(section):
                dataset_params[option] = config.get(section, option)
        #construct net_paramss
        elif section == 'Net':
            for option in config.options(section):
                net_params[option] = config.get(section, option)
        #construct solver_params
        elif section == 'Solver':
            for option in config.options(section):
                solver_params[option] = config.get(section, option)
    return common_params, dataset_params, net_params, solver_params