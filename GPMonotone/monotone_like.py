#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:29:08 2018

@author: nimishawalgaonkar
"""

from gpflow.likelihoods import Likelihood
from gpflow.likelihoods import probit
from gpflow.param import Param
from gpflow import densities
from gpflow import transforms
from gpflow.param import AutoFlow
from gpflow._settings import settings
float_type = settings.dtypes.float_type
import tensorflow as tf

class MonotoneLikelihood(Likelihood):
    def __init__(self):
        """
        Likelihood for Gaussian Process with monotonicity constraints
        """
        Likelihood.__init__(self)
        self.nu = 1e-6
        self.noise_variance = Param(1.0, transforms.positive)
    
    def logp_ygf(self, F, Y):
        return tf.reduce_sum(densities.gaussian(F, Y, self.noise_variance))
    
    def logp_m(self, F_prime, values, invlink = probit):
        """
        Values would be 0 if the derivatives are negative 
        Values would be 1 if the derivatives are positive
        """
        return tf.reduce_sum(densities.bernoulli(invlink(1./self.nu*F_prime), values))
    
    def logp(self, F, Y, F_prime, ones):
        log_like_ygp = self.logp_ygf(F, Y)
        log_like_m = self.logp_m(F_prime, ones, invlink = probit)
        log_like = log_like_ygp + log_like_m
        #print 'we are here ...'
        #log_like = log_like_ygp
        return log_like 
    
    @AutoFlow((float_type, [None, None]),
              (float_type, [None, None]),
              (float_type, [None, None]),
              (float_type, [None, None]))
    
    def compute_logp(self, F, Y, F_prime, ones):
        return self.logp(F,Y, F_prime, ones)
    
    
if __name__ == '__main__':
    import numpy as np
    f = np.array([1,2,3])[:,None]
    f_prime = np.array([4,5])[:,None]
    ones = np.ones(f_prime.shape[0])[:,None]
    y = np.array([1,2,3])[:,None]
    
    L = MonotoneLikelihood()
    
    
    