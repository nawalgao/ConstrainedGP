#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 19:00:55 2018

@author: nimishawalgaonkar
"""

import numpy as np
import tensorflow as tf
from gpflow.priors import Gaussian
from gpflow.param import AutoFlow, DataHolder, Param
from gpflow._settings import settings
float_type = settings.dtypes.float_type
from .monotone_gp import MonotoneGP
from .monotone_conditional import monotone_conditional

class MonotoneGPMC(MonotoneGP):
    def __init__(self, X, Y, X_prime, values):
        """
        X is a data vector, size N x 1
        X_prime is a vector, size M x 1
        Y is a data vector, size N x 1 
        values is a vector, size M x 1
        values are 0 when derivatives of function are negative 
        values are 1 when derivatives are positive
    
        This is a vanilla implementation of a GP with monotonicity contraints and HMC sampling
        Refer:
        https://bayesopt.github.io/papers/2017/9.pdf
        """
        X_concat = np.vstack([X, X_prime])
        X_concat = DataHolder(X_concat, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        MonotoneGP.__init__(self, X_concat, Y)
        self.X = DataHolder(X)
        self.values = DataHolder(values)
        self.X_prime = DataHolder(X_prime)
        self.num_data = X_concat.shape[0]
        self.num_x_points = X.shape[0]
        self.num_der_points = X_prime.shape[0]
        self.num_latent = Y.shape[1]
        self.V = Param(np.zeros((self.num_data, self.num_latent)))
        self.V.prior = Gaussian(0., 1.)
    def compile(self, session = None, graph = None, optimizer = None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X_concat.shape[0]:
            #print "wink wink"
            self.num_data = self.X_concat.shape[0]
            self.V = Param(np.zeros((self.num_data, self.num_latent)))
            self.V.prior = Gaussian(0., 1.)
        
        return super(MonotoneGPMC, self).compile(session = session,
                                                 graph = graph,
                                                 optimizer = optimizer)
    
    def build_likelihood(self):
        #print '1111...'
        Kjoint = self.kern.Kj(self.X, self.X_prime)
        L = tf.cholesky(Kjoint +
                        tf.eye(tf.shape(self.X_concat)[0], dtype=float_type)*
                        settings.numerics.jitter_level)
        #print '2222...'
        F_concat = tf.matmul(L, self.V)
        F, F_prime = tf.split(F_concat, [self.num_x_points, self.num_der_points])
        log_like = self.likelihood.logp(F, self.Y, F_prime, self.values)
        return log_like
    
    def build_predict(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        
        #V, V_prime = tf.split(self.V, [self.num_x_points, self.num_der_points])
        mu, var = monotone_conditional(Xnew, self.X,self.X_prime,
                                           self.kern, self.V, whiten=True)
        return mu, var

