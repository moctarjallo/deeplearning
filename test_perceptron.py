import unittest
import numpy as np
from perceptron import Perceptron

class TestInitialization(unittest.TestCase):
    def test_with_given_weights(self):
        W = np.array([-1, -2])
        p = Perceptron(weights=W)
        
        self.assertTrue(np.array_equal(p.weights, np.array([-1, -2])))
        
    def test_with_random_weights(self):
        X = np.array([1, 2])
        p = Perceptron(shape=X.shape)
        
        self.assertEqual(p.weights.shape, (2,))
        
    def test_null_inputs(self):
        self.assertRaises(TypeError, Perceptron)
        
        
class TestForward(unittest.TestCase):
    def test_(self):
        X = np.array([1, 2])
        W = np.array([-1, -2])
        p = Perceptron(weights=W)
        
        y = 1*(-1) + 2*(-2)
        self.assertEqual(p.forward(X), y)
        
    def test_call(self):
        X = np.array([1, 2])
        W = np.array([-1, -2])
        p = Perceptron(weights=W)
        
        y = 1*(-1) + 2*(-2)
        self.assertEqual(p(X), y)
        
        
if __name__ == '__main__':
    unittest.main()