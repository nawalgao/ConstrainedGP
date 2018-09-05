#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:49:21 2018

@author: nimishawalgaonkar
"""

import numpy as np
import tensorflow as tf
from gpflow.priors import Gaussian
from gpflow.param import AutoFlow, DataHolder, Param
from gpflow._settings import settings
float_type = settings.dtypes.float_type
from .unimodal_gp import UnimodalGP
from .unimodal_conditional import monotone_conditional
from .unimodal_like import UnimodalPrefLikelihood

class UnimodalGPMC(UnimodalGP):
    def __init__(self, X, Y, X_prime):
        """
        X is a data vector, size N x 1
        X_prime is a vector, size M x 1
        Y is a data vector, size N x 1 

        This is a vanilla implementation of a GP with unimodality contraints and HMC sampling
        Refer:
        https://bayesopt.github.io/papers/2017/9.pdf
        """
        X_concat = np.vstack([X, X_prime])
        X_concat = DataHolder(X_concat, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        UnimodalGP.__init__(self, X_concat, Y)
        self.X = DataHolder(X)
        self.X_prime = DataHolder(X_prime)
        self.num_data = X_concat.shape[0]
        self.num_x_points = X.shape[0]
        self.num_der_points = X_prime.shape[0]
        self.num_latent = Y.shape[1]
        
        self.Vf = Param(np.zeros((self.num_data, self.num_latent)))
        self.Vf.prior = Gaussian(0., 1.)
        
        self.Vg = Param(np.zeros((2*self.num_der_points, self.num_latent)))
        self.Vg.prior = Gaussian(0., 1.)
        
    
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
            self.num_x_points = self.X.shape[0]
            self.num_der_points = self.X_prime.shape[0]
            self.Vf = Param(np.zeros((self.num_data, self.num_latent)))
            self.Vf.prior = Gaussian(0., 1.)
            self.Vg = Param(np.zeros((2*self.num_der_points, self.num_latent)))
            self.Vg.prior = Gaussian(0., 1.)
        
        return super(UnimodalGPMC, self).compile(session = session,
                                                 graph = graph,
                                                 optimizer = optimizer)
    def build_likelihood(self):
        Kfjoint = self.kern_f.Kj(self.X, self.X_prime)
        Kgjoint = self.kern_g.Kj(self.X_prime, self.X_prime)
        
        Lf = tf.cholesky(Kfjoint + tf.eye(tf.shape(self.X_concat)[0], dtype=float_type)*
                        settings.numerics.jitter_level)
        Lg = tf.cholesky(Kgjoint + tf.eye(2*tf.shape(self.X_prime)[0], dtype=float_type)*
                        settings.numerics.jitter_level)
        
        F_concat = tf.matmul(Lf, self.Vf)
        F, F_prime = tf.split(F_concat, [self.num_x_points, self.num_der_points])
        
        G_concat = tf.matmul(Lg, self.Vg)
        G, G_prime = tf.split(G_concat, [self.num_der_points, self.num_der_points])
        log_like = self.likelihood.logp(self.Y, F, F_prime, G, G_prime)
        return log_like
    
    def build_predict(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        
        #V, V_prime = tf.split(self.V, [self.num_x_points, self.num_der_points])
        mu_f, var_f = monotone_conditional(Xnew, self.X,self.X_prime, self.kern_f, self.Vf, 
                                       whiten=True)
        return mu_f, var_f
    
    def build_predict_g(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(G* | (G=LgVg) )

        where G* are points on the GP at Xnew, G=LgVg are points on the GP at X.

        """
        
        #V, V_prime = tf.split(self.V, [self.num_x_points, self.num_der_points])
        mu_g, var_g = monotone_conditional(Xnew, self.X_prime,self.X_prime,
                                           self.kern_g, self.Vg, 
                                       whiten=True)
        return mu_g, var_g

class UnimodalPrefGPMC(UnimodalGPMC):
    def __init__(self, X, Y, X_prime):
        """
        X is a data vector, size 2N x 1
        X_prime is a vector, size M x 1
        Y is a data vector, size N x 1 consisting of ones and zeros
        y_i = 1 : current is preferred over previous
        y_i = 0 : previous is preferred over current

        This is a vanilla implementation of a GP preference model with 
        unimodality contraints and HMC sampling
        Refer:
        https://bayesopt.github.io/papers/2017/9.pdf
        """
        
        UnimodalGPMC.__init__(self, X, Y, X_prime)
        self.likelihood = UnimodalPrefLikelihood()
    
    def build_likelihood(self):
        Kfjoint = self.kern_f.Kj(self.X, self.X_prime)
        Kgjoint = self.kern_g.Kj(self.X_prime, self.X_prime)
        
        Lf = tf.cholesky(Kfjoint + tf.eye(tf.shape(self.X_concat)[0], dtype=float_type)*
                        settings.numerics.jitter_level)
        Lg = tf.cholesky(Kgjoint + tf.eye(2*tf.shape(self.X_prime)[0], dtype=float_type)*
                        settings.numerics.jitter_level)
        
        F_concat = tf.matmul(Lf, self.Vf)
        F, F_prime = tf.split(F_concat, [self.num_x_points, self.num_der_points])
        
        G_concat = tf.matmul(Lg, self.Vg)
        G, G_prime = tf.split(G_concat, [self.num_der_points, self.num_der_points])
        log_like = self.likelihood.logp(self.Y, F, F_prime, G, G_prime)
        return log_like
        
    
        