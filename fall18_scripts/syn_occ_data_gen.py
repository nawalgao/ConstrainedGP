#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:29:52 2018

@author: nimishawalgaonkar
"""

import sys
sys.path.append('../')

from GPFlowPref.SynOccupant import datagen

config_file = '../GPFlowPref/config_files/thermal_config.json'
save_file_name1 = '../GPFlowPref/data/initial_duels/train1D.npz'
save_file_name2 = '../GPFlowPref/data/initial_duels/train2D.npz'
ThermalP = datagen.ThermalPrefDataGen(config_file)
    
X1, y_pr1 = ThermalP.pairwise1D(2, save_file_name1, save_file = False)
X2, y_pr2 = ThermalP.pairwise2D(40, save_file_name2, save_file = False)