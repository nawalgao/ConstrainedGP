#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:13:12 2018

@author: nimishawalgaonkar
"""

import tensorflow as tf
import numpy as np
from gpflow.param import Param, ParamList, Parameterized, AutoFlow
from gpflow._settings import settings
from gpflow.mean_functions import MeanFunction
from gpflow import transforms
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64



class Linear(MeanFunction):
    """
    y_i = A x_i + b
    """
    def __init__(self, A=None, b=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.

        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        A = np.ones((1, 1)) if A is None else A
        b = np.zeros(1) if b is None else b
        MeanFunction.__init__(self)
        self.A = Param(np.atleast_2d(A))
        self.b = Param(b)

    def __call__(self, X):
        return tf.matmul(X, self.A) + self.b
    
    


class Parabola(MeanFunction):
    """
    y_i = -1/2* A (x - mu)^2
    Only available for 1D features for now.
    
    """
    def __init__(self, A = None, mu = None):
        A = np.ones(1) if A is None else A
        mu = np.zeros(1) if mu is None else mu
        MeanFunction.__init__(self)
        self.A = Param(A)
        self.mu = Param(mu)
        
    def __call__(self, X):
        diff = X - self.mu
        sqr = tf.square(diff)
        mm = -1./2*self.A*sqr
        
        return mm
    
    @AutoFlow((float_type, [None, None]))
    def compute_mean(self, X):
        return self.__call__(X)
    
#self.variance * tf.exp(-self.square_dist(X, X2) / 2)

class Normal(MeanFunction):
    """
    1D Normal distribution 
    y_i = v*exp(-1/2l^2 * (x - mu)^2)
    """
    def __init__(self, mu = None, lengthscale = None, signal_variance = None):
        mu = np.zeros(1) if mu is None else mu
        lengthscale = 0.2*np.ones(1) if lengthscale is None else lengthscale
        signal_variance = 3*np.ones(1) if signal_variance is None else signal_variance
        MeanFunction.__init__(self)
        self.signal_variance = Param(signal_variance, transforms.positive)
        self.lengthscale = Param(lengthscale, transforms.positive)
        self.mu = Param(mu)
        
    def __call__(self, X):
        diff = X - self.mu
        sqr = tf.square(diff)
        return self.signal_variance * tf.exp(-1./2 * sqr)
    
    @AutoFlow((float_type, [None, None]))
    def compute_mean(self, X):
        return self.__call__(X)

class Zero(MeanFunction):
    def __call__(self, X):
        return tf.zeros(tf.stack([tf.shape(X)[0], 1]), dtype=float_type)
    @AutoFlow((float_type, [None, None]))
    def compute_mean(self, X):
        return self.__call__(X)
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    import sys
    sys.path.append('../')
    from SynOccupant import datagen
    
    config_file = '../config_files/thermal_config.json'
    
    x = np.linspace(15,27,100)[:, None]
    
    TDG = datagen.ThermalPrefDataGen(config_file)
    
    xnorm = TDG.normalize1D(x)
    
    mu = 27
    munorm = datagen.ThermalPrefDataGen(config_file).normalize1D(mu)
    
    A = 6.
    y = -1./2*A*(xnorm - munorm)**2
    
    plt.plot(xnorm,y)
    #plt.ylim(0, 5)
    