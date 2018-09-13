#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Alejandro Debus"
__email__ = "aledebus@gmail.com"

'''
In this module are the loss functions and their derivatives.
'''

import numpy as np

def cross_entropy(y, y_hat):
    '''
    For classification
    '''

    m = y.shape[1]

    cost = (-1 / m) * np.sum(np.multiply(y, np.log(y_hat)) + np.multiply((1 - y), np.log(1 - y_hat)))

    cost = np.squeeze(cost)

    return cost

def cross_entropy_derivative():
    pass

def mean_squared_error(y, y_hat):
    '''
    For regression
    '''

    cost = 0.5 * np.sum(np.square(y - y_hat))

    return cost

def mean_squared_error_derivative():
    pass
