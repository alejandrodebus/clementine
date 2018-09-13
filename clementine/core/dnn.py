#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Alejandro Debus"
__email__ = "aledebus@gmail.com"

'''
In this module is the Deep Neural Network
'''

import numpy as np
from clementine.core.activations import sigmoid, sigmoid_derivative, relu, relu_derivative
from clementine.core.loss import cross_entropy


class DNN():

    def __init__(self, size_layers):
        self.size_layers = size_layers


    def __initialize_parameters(self, size_layers, init_weights = 'random', seed = 123):
        '''


        Arguments:

        Return:
        '''

        np.random.seed(seed)

        num_layers = len(size_layers)

        parameters = {}

        for layer in range(1, num_layers):
            if init_weights == 'random':
                parameters['weights' + str(layer)] = np.random.randn(size_layers[layer], size_layers[layer - 1]) * 0.01
            elif init_weights == 'heetal':
                parameters['weights' + str(layer)] = np.random.randn(size_layers[layer], size_layers[layer - 1]) * np.sqrt (2 / size_layers[layer - 1])
            elif init_weights == 'xavier':
                parameters['weights' + str(layer)] = np.random.randn(size_layers[layer], size_layers[layer - 1]) * np.sqrt (1 / size_layers[layer - 1])

            parameters['biases' + str(layer)] = np.zeros((size_layers[layer], 1))

        return parameters

    def __forward_propagation(self, X, parameters):

        num_layers = len(parameters) // 2

        A = X

        linear_output_cache = []

        for layer in range(1, num_layers):

            Z = np.dot(parameters['weights' + str(layer)], A) + parameters['biases' + str(layer)]
            A = relu(Z)
            linear_output_cache.append((Z, parameters['weights' + str(layer)], parameters['biases' + str(layer)]))

        # Last layer
        Z = np.dot(parameters['weights' + str(num_layers)], A) + parameters['biases' + str(num_layers)]
        A = sigmoid(Z)
        linear_output_cache.append((Z, parameters['weights' + str(num_layers)], parameters['biases' + str(num_layers)]))

        return A, linear_output_cache

    def __backward_propagation(self, A, Y, linear_output_cache):


        num_layers = len(linear_output_cache)

        gradients = {}

        # derivative of loss function respect with output layer
        dL_da = - np.divide(Y, A) + np.divide(1 - Y, 1 - A)

        cache = linear_output_cache[-1]

        m = cache[0].shape[1]

        Z = cache[0]
        dZ = dL_da * sigmoid_derivative(Z)

        gradients['dA' + str(num_layers)] = np.dot(cache[1].T, dZ)
        gradients['dW' + str(num_layers)] = (1 / m) * np.dot(dZ, cache[0].T)
        gradients['db' + str(num_layers)] = (1 / m) * np.sum(dZ, axis = 1, keepdims = True)

        dA_prev = dL_da

        for layer in reversed(range(num_layers - 1)):

            cache = linear_output_cache[layer]
            Z = cache[0]
            dZ = dA_prev * relu_derivative(Z)
            gradients['dA' + str(layer + 1)] = np.dot(cache[1].T, dZ)
            gradients['dW' + str(layer + 1)] = (1 / m) * np.dot(dZ, cache[0].T)
            gradients['db' + str(layer + 1)] = (1 / m) * np.sum(dZ, axis = 1)

            W = cache[1].T
            dA_prev = np.dot(W, dZ)

        return gradients

    def __update_parameters(self, parameters, gradients, learning_rate):

        num_layers = len(parameters) // 2

        for layer in range(num_layers):
            parameters['weights' + str(layer + 1)] -= learning_rate * gradients['dW' + str(layer + 1)]
            parameters['biases' + str(layer + 1)] -= learning_rate * gradients['db' + str(layer + 1)]

        return parameters

    def train(self, X, y, learning_rate, iterations, log_interval = 0):

        parameters = self.__initialize_parameters(self.size_layers)

        for iter in range(0, iterations):

            a_l, linear_output_cache = self.__forward_propagation(X, parameters)

            loss = cross_entropy(a_l, y)

            gradients = self.__backward_propagation(a_l, y, linear_output_cache)

            parameters = self.__update_parameters(parameters, gradients, learning_rate)

            if (log_interval > 0 and (iter % log_interval == 0)):
                print("Iteration: {}/{} - Loss: {}".format(iter, iterations, loss))

        return parameters

    def test(self, X, y, parameters):

        m = X.shape[1]

        predictions = np.zeros((1, m))

        probabilities, caches = self.__forward_propagation(X, parameters)

        for i in range(0, probas.shape[1]):
            if (probabilities[0, i] > 0.5):
                predictions[0, i] = 1
            else:
                predictions[0, i] = 0

        print('Accuracy: '  + str(np.sum((p == y) / m)))

        return predictions
