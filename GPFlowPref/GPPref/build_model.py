# Copyright 2016 James Hensman, alexggmatthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
from gpflow.gpmc import GPMC
from gpflow._settings import settings
float_type = settings.dtypes.float_type


class GpPrefLearningHMC(GPMC):
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None, num_latent=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        This is a vanilla implementation of a GP with a non-Gaussian
        likelihood. The latent function values are represented by centered
        (whitened) variables, so

            v ~ N(0, I)
            f = Lv + m(x)

        with

            L L^T = K

        """
        
        GPMC.__init__(self, X, Y, kern, likelihood, mean_function)
    
    def build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of a general GP
        model.

            \log p(Y, V | theta).

        """
        K = self.kern.K(self.X)
        L = tf.cholesky(K + tf.eye(tf.shape(self.X)[0], dtype=float_type)*settings.numerics.jitter_level)
        F = tf.matmul(L, self.V) + self.mean_function(self.X)
        
        F1,F2 = tf.split(F, num_or_size_splits=2)
        F_diff = tf.subtract(F1,F2)
        return tf.reduce_sum(self.likelihood.logp(F_diff, self.Y))