#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Alejandro Debus"
__email__ = "aledebus@gmail.com"

'''
In this module is padding for CNN
'''

import numpy as np

def padding(X, pad = 0):
    '''
    Returns matrix X with zero padding

    Arguments:
    X: numpy array (matrix)
    pad: padding width

    Return:
    X_padding: matrix X with padding
    '''

    X_padding = np.pad(X, pad, 'constant', constant_values = 0)

    return X_padding
