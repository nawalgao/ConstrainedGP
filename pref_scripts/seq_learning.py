#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:44:42 2018

@author: nimishawalgaonkar
"""

import sys
sys.path.append('../')
import numpy as np

from GPFlowPref.GPElicit.elicit import seq_learning
from GPFlowPref.SynOccupant import datagen

config_file = '../GPFlowPref/config_files/thermal_config.json'
save_file_name = '../GPFlowPref/data/initial_duels/train1D.npz'


data = np.load(save_file_name)
V = datagen.ThermalPrefDataGen(config_file)
X = data['X']
Y = data['Y']

budget = 15
trial_num = 6
model_num = 1
reachable = False
savefig = True
mcmc = True

Aq, X, Y = seq_learning(X, Y, budget,config_file,
             trial_num, model_num, mcmc,
             reachable, savefig)

