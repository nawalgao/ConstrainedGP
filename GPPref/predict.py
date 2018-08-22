#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 14:20:13 2017

@author: nawalgao
"""

import numpy as np

class Predict(object):
    """
    Preference Model Prediction class
    """
    
    def __init__(self, m, samples):
        self.m = m
        self.samples = samples
        
    def u(self, Xnorm):
        """
        Utility posterior function values for normalized state matrix X_norm
        Inputs:
            Xnorm (N X 1) when only 1D (VI) feature vector or
            Xnorm (N X 2) when 2D (VI + SP) feature matrix
        Ouput:
            Posteior mean and variance of GP (for each hyperparameter sample)
        """
        meanmat = np.zeros(shape = (self.samples.shape[0], Xnorm.shape[0]))
        varmat = np.zeros(shape = (self.samples.shape[0], Xnorm.shape[0]))
        for i, s in enumerate(self.samples):
            self.m.set_state(s)
            mean, var = self.m.predict_f(Xnorm)
            meanmat[i,:] = mean[:,0]
            varmat[i,:] = var[:,0]
        return meanmat, varmat
    
    def u_test_train(self, Xtrainnorm, Xtestnorm):
        """
        Posterior GP mean and variance at training and testing state
        for each hyperparameter sample
        Inputs:
            Xtrainnorm : Normalized training state values
            Xtestnorm : Normalized testing state values
        Outputs:
            mtrainmat : GP mean values at training points
            vartrainmat : Variance of GP at training points
            mtestmat : GP mean at testing points | GP at training points
            vartestmat : Variance of GP at testing points | GP at training points
        """
        num_feat = Xtrainnorm.shape[1]/2
        Xtrain_seccol_norm = Xtrainnorm[:,num_feat:]
        mtrainmat, vartrainmat = self.u(Xtrain_seccol_norm)
        mtestmat, vartestmat = self.u(Xtestnorm)
        
        return mtrainmat, vartrainmat, mtestmat, vartestmat 