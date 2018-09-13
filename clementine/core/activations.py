#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Alejandro Debus"
__email__ = "aledebus@gmail.com"

'''
In this module are the activation functions and their derivatives.
'''

import numpy as np

def sigmoid(x):
    '''
    Calculate the sigmoid of x

    Arguments:
    x: a scalar or numpy array

    Return:
    s: sigmoid of x (a scalar or numpy array)
    '''

    s = 1. / (1 + np.exp(-x))

    return s

def sigmoid_derivative(x):
    '''
    Calculate the derivative of the sigmoid

    Arguments:
    x: a scalar or numpy array

    Return:
    s: derivative of sigmoid (a scalar or numpy array)
    '''

    s = sigmoid(x) * (1 - sigmoid(x))

    return s

def relu(x):
    '''
    Calculate the ReLU of x

    Arguments:
    x: a scalar or numpy array

    Return:
    s: ReLU of x (a scalar or numpy array)
    '''

    s = np.maximum(0,x)

    return s

def relu_derivative(x):
    '''
    Calculate the derivative of the ReLU

    Arguments:
    x: a scalar or numpy array

    Return:
    s: derivative of ReLU (a scalar or numpy array)
    '''

    s = np.greater(x, 0).astype(int)

    return s
