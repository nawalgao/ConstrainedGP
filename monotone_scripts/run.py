#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 11:37:53 2018

@author: nimishawalgaonkar
"""

import gpflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk", font_scale = 1.4)

X = np.linspace(0.,1, 100)[:,None]
X_prime = np.linspace(0, 1, 50)[:,None]
Y = 5*X

#plt.plot(X, Y)


import sys
sys.path.append('../')
from GPMonotone.ker_w_der import ExtendRBF1D
from GPMonotone.monotone_gpmc import MonotoneGPMC

# Define kernel for monotonicity
K = ExtendRBF1D()

# Define model configuration
m = MonotoneGPMC(X, Y, X_prime)
m.likelihood.noise_variance = 0.01
m.likelihood.noise_variance.fixed = True

# Priors over hyperparameters of interest
m.kern.lengthscale.prior = gpflow
m.kern.lengthscale.fixed = True
m.kern.signal_variance = 1
m.kern.signal_variance.fixed = True

print 'Log like is '
print m.compute_log_likelihood()