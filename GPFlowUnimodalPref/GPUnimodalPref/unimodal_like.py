#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 08:53:19 2018

@author: nimishawalgaonkar
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:29:08 2018

@author: nimishawalgaonkar
"""

import numpy as np
from gpflow.likelihoods import Likelihood
from gpflow.likelihoods import probit
from gpflow.param import Param
from gpflow import densities
from gpflow import transforms
from gpflow.param import AutoFlow
from gpflow._settings import settings
float_type = settings.dtypes.float_type
import tensorflow as tf


#def probit(x):
#    return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1 - 2e-3) + 1e-3

class UnimodalLikelihood(Likelihood):
    def __init__(self):
        """
        Likelihood for Gaussian Process with unimodality constraints
        """
        Likelihood.__init__(self)
        self.nuf = 1./1e-6
        self.nug = 1./1e-6
        self.noise_variance = Param(1.0, transforms.positive)
    
    def logp_ygivenf(self, F, Y):
        
        return tf.reduce_sum(densities.gaussian(F, Y, self.noise_variance))
    
    def log_interlike(self, F_prime, G):
        """
        Refer to page 2
        https://bayesopt.github.io/papers/2017/9.pdf
        """
        prob_neg_f_prime = probit(-1*self.nuf*F_prime)
        prob_neg_g = probit(-1*G)
        prob_pos_f_prime = probit(self.nuf*F_prime)
        prob_pos_g = probit(G)
        
        prod1 = prob_neg_f_prime*prob_neg_g
        prod2 = prob_pos_f_prime*prob_pos_g
        summation = prod1 + prod2
        log = tf.log(summation)
        
        return tf.reduce_sum(log)
    
    def log_monotonic(self, G_prime):
        """
        Refer to page 2
        https://bayesopt.github.io/papers/2017/9.pdf
        """
        prob_gprime = probit(-self.nug*G_prime)
        log_prob = tf.log(prob_gprime)
        
        return tf.reduce_sum(log_prob)
    
    def logp(self, Y, F, F_prime, G, G_prime):
        """
        Refer to page 2
        https://bayesopt.github.io/papers/2017/9.pdf
        """
        log_like1 = self.logp_ygivenf(F, Y)
        log_like2 = self.log_interlike(F_prime, G)
        log_like3 = self.log_monotonic(G_prime)
        log_like = log_like1 + log_like2 + log_like3
        
        return log_like
    
    @AutoFlow((float_type, [None, None]),
              (float_type, [None, None]),
              (float_type, [None, None]),
              (float_type, [None, None]),
              (float_type, [None, None]))
    
    def compute_logp(self, Y, F, F_prime, G, G_prime):
        return self.logp(Y, F, F_prime, G, G_prime)

class UnimodalPrefLikelihood(UnimodalLikelihood):
    def __init__(self):
        """
        Likelihood for Gaussian Process Preference Learning model with unimodality constraints
        """
        UnimodalLikelihood.__init__(self)
    def logp_ygivenf(self, F, Y, invlink = probit):
        F1, F2 = tf.split(F, num_or_size_splits=2)
        F_diff = tf.subtract(F1,F2)
        #Fn = F_diff/(np.sqrt(2)*tf.sqrt(self.noise_variance))
        Fn = F_diff
        return tf.reduce_sum(densities.bernoulli(invlink(Fn), Y))
        
    
    
    
if __name__ == '__main__':
    f = np.array([0,1,2,3])[:,None]
    f_prime = np.array([4,5])[:,None]
    g = np.array([6,7])[:,None]
    g_prime = np.array([8,9])[:,None]
    y = np.array([0.,1.])[:,None]
    
    L = UnimodalPrefLikelihood()
    L.compute_logp(y,f,f_prime,g,g_prime)
        
        