"""
A Perceptron is no other than a linear function of a set of inputs X and 
an output Y
Consider a table like this one:
    |x1 | x2| y |
   --------------
   | 1 | 2 | 0 |
  | 1 | 0 | 1 |
One's goal would be to find the relationship between the inputs x1, x2
and the output y.
To do so one could use a set of parameters called weights, which, linearly
combined with inputs x1, x2, give the correct output y. The problems then
boils down to finding such correct weights.
This is where the model of the Perceptron comes in.
So let's model this using a python class first of all and some numpy functions.
"""

import numpy as np

class Perceptron:
    # A Perceptron is defined as simple as a set of weights represented
    # by a numpy array. If weights are not given by user, then he must
    # provide a shape to initialize random weights of that shape.
    def __init__(self, weights=np.array([]), shape=()):
        if weights.any():
            self.weights = weights
        elif not shape:
            raise TypeError("Must provide a shape for the Perceptron")
        else:
            self.weights = np.random.rand(*shape)
        
    def forward(self, x):
        y = x.dot(self.weights)
        return y
    
    # just to recall that this is no other than a model of a function
    def __call__(self, x):
        return self.forward(x)