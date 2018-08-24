#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:21:20 2018

@author: nimishawalgaonkar
"""

import sys
sys.path.append('../')
import numpy as np

from GPFlowPref.SynOccupant import datagen
from GPFlowPref.GPPref.train import Train
from GPFlowPref.GPPref.predict import Predict

config_file = '../GPFlowPref/config_files/thermal_config.json'
save_file_name = '../GPFlowPref/data/initial_duels/train1D.npz'

data = np.load(save_file_name)
V = datagen.ThermalPrefDataGen(config_file)
X = data['X']
Y = data['Y']
Xnorm = V.normalize1Dpairwise(X)

# Train  
TRAIN = Train(Xnorm, Y, config_file)
m, samples = TRAIN.mcmc(3)
m.optimize(maxiter = 20)

# Test
PREDICT = Predict(m)
meanmat, varmat = PREDICT.u(samples, Xnorm)

