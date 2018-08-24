#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Trains a Preference model

"""


import numpy as np
import json # Needed for config file
import os
import gpflow
 
from . import build_model
from . import mean_func

class Train(object):
    """
    Train the preference learning model using the pairwise comparisons
    Train a GP Preference Learning Model
    Inputs:
        X : previous feat columnwise contatenated with current feat X = [X_pre, X_current]
        Y : binary variable indicating whether current or previous state is preferred 
        Y = 1 if current is prefered ; 0 if previous is preferred 
        config_file : visual preferences related configuration settings  
    """
    
    def __init__(self, X, Y, config_file, model_num = 3):
        self.X = X
        self.Y = Y.astype(float)
        self.config_file = config_file
        
        # sanity check
        assert os.path.isfile(self.config_file)
        num_feat = self.X.shape[1]/2
        Xn = np.vstack([self.X[:,num_feat:], self.X[:,:num_feat]])
         # Read configuration file
        with open(self.config_file, 'r') as fd:
            config = json.loads(fd.read())
        
         # Priors
        lengthscale_prior = eval(config['Prior']['lengthscales'])
        variance_prior = eval(config['Prior']['variance'])
        para_mean_mu_prior = eval(config['Prior']['parabola_norm_mu'])
        para_mean_A_prior = eval(config['Prior']['parabola_A'])
        
        l1_prior = eval(config['Prior']['l1'])
        l2_prior =  eval(config['Prior']['l2'])
        v1_prior =  eval(config['Prior']['v1'])
        v2_prior = eval(config['Prior']['v2'])
        mu_prior = eval(config['Prior']['mu'])
        
        
        # Kernel
        k = gpflow.kernels.RBF(input_dim = num_feat, ARD = True)
        # Likelihood
        l = gpflow.likelihoods.Bernoulli()
        
        if model_num == 1:
            print 'model is 1'
            print 'No mean, Gamma prior on signal var and lengthscale'
            
            # Model definition
            m = build_model.GpPrefLearningHMC(Xn, self.Y, k, l)
            m.kern.lengthscales.prior = lengthscale_prior
            m.kern.variance.prior = variance_prior
        
        if model_num == 2:
            print 'model is 2'
            print 'No mean, fix lengthscale = 0.2, signal variance = 10'
            # Model definition
            m = build_model.GpPrefLearningHMC(Xn, self.Y, k, l)
            m.kern.lengthscales = config['Hyper_Config']['l']
            m.kern.lengthscales.fixed = True
            m.kern.variance = config['Hyper_Config']['v']
            m.kern.variance.fixed = True
        
        if model_num == 3:
            print 'model is 3'
            print 'with mean, lengthscale fix = 0.2, var fix = 0.1'
            meanf = mean_func.Parabola(1,0)
            # Model definition
            m = build_model.GpPrefLearningHMC(Xn, self.Y, k, l, meanf)
            m.kern.lengthscales = config['Hyper_Config']['l']
            m.kern.lengthscales.fixed = True
            m.kern.variance = config['Hyper_Config']['v']
            m.kern.variance.fixed = True
            m.mean_function.A.prior = para_mean_A_prior
            m.mean_function.mu.prior = para_mean_mu_prior
            
        if model_num == 4:
            print 'model is 4'
            print 'with mean, var gamma prior, fix lengthscale = 0.2'
            meanf = mean_func.Parabola(1,0)
            # Model definition
            m = build_model.GpPrefLearningHMC(Xn, self.Y, k, l, meanf)
            m.kern.lengthscales = config['Hyper_Config']['l']
            m.kern.lengthscales.fixed = True
            m.kern.variance.prior = variance_prior
            m.mean_function.A.prior = para_mean_A_prior
            m.mean_function.mu.prior = para_mean_mu_prior
        
        if model_num == 5:
            print 'model is 5'
            print 'Exponential (normal) mean func'
            print 'l1 and l2 lognormal'
            print 'v1 lognormal'
            print 'v2 Gaussian'
            meanf = mean_func.Normal()
            # Model definition
            m = build_model.GpPrefLearningHMC(Xn, self.Y, k, l, meanf)
            m.kern.lengthscales.prior = l2_prior
            m.kern.variance.prior = v2_prior
            m.mean_function.lengthscale.prior = l1_prior
            m.mean_function.signal_variance.prior = v1_prior
            m.mean_function.mu.prior = mu_prior
        
        self._m = m
        
    def mcmc(self, config_file):
        
        with open(config_file, 'r') as fd:
            config = json.loads(fd.read())
        
        
        # HMC sampling
        MAP_optimize_maxiter = config['MCMC']['MAP_optimize_maxiter']
        num_samples = config['MCMC']['num_samples']
        thin = config['MCMC']['thin']
        burn = config['MCMC']['burn']
        epsilon = config['MCMC']['epsilon']
        Lmax = config['MCMC']['Lmax']
        verbose = eval(config['MCMC']['verbose'])
        
        self._m.optimize(maxiter= MAP_optimize_maxiter) # start near MAP
        samples = self._m.sample(num_samples, verbose= verbose,
                           epsilon= epsilon, thin = thin, burn = burn, Lmax=Lmax)
        
        return self._m, samples
    
    def maxpost(self, config_file):
        with open(config_file, 'r') as fd:
            config = json.loads(fd.read())
        
        # MAP settings
        max_iter =  config['MAP']['max_iter']
        self._m.optimize(maxiter= max_iter) # start near MAP
        
        return self._m