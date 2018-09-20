#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:37:07 2018

@author: nimishawalgaonkar
"""

import pickle

def save_object(obj, filename):
    """
	saves gpflow model object
	"""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def read_pickle_object(pickle_file_name):
    """
    Reads gpflow object m saved as a pickle file
    """
    with open(pickle_file_name, 'rb') as inn:
        m = pickle.load(inn)
    return m

