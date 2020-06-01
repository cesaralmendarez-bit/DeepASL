import numpy as np
import pickle
import os

class Softmax2:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
            # We divide by input_len to reduce the variance of our initial values
        if os.path.getsize('NominalWeights.txt') > 0:
            with open('NominalWeights.txt', 'rb') as f:
                self.weights = pickle.load(f)
        if os.path.getsize('NominalBiases.txt') > 0:
            with open('NominalBiases.txt', 'rb') as t:
                self.biases = pickle.load(t)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''

        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)
