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
        b = .5
        p = Perceptron(weights=W, bias=.5)
        
        y = 1*(-1) + 2*(-2) + b
        self.assertEqual(p.forward(X), p.activation(y))
        
    def test_call(self):
        X = np.array([1, 2])
        W = np.array([-1, -2])
        b = 1
        p = Perceptron(weights=W, bias=b)
        
        y = 1*(-1) + 2*(-2) + b
        self.assertEqual(p(X), p.activation(y))
        
class TestUpdate(unittest.TestCase):
    # This is equivalent to `test_perceptron_update1()` from the problem statement
    def test_1(self):
        x = np.array([0,1])
        y = -1
        w = np.array([1,1])
        p = Perceptron(weights=w)
        p.update(x,y)
        #breakpoint()
        self.assertTrue((p.weights.reshape(-1,) == np.array([1,0])).all())
        
    def test_2(self):
        # This is equivalent to `test_perceptron_update2()` from the problem statement
        x = np.random.rand(25)
        y = 1
        w = np.zeros(25)
        p = Perceptron(weights=w)
        p.update(x,y)
        self.assertTrue(np.linalg.norm(p.weights-x)<1e-8)
        
    def test_3(self):
        # This is equivalent to `test_perceptron_update3()` from the problem statement
        x = np.random.rand(25)
        y = -1
        w = np.zeros(25)
        p = Perceptron(weights=w)
        p.update(x,y)
        self.assertTrue(np.linalg.norm(p.weights+x)<1e-8)
        
        
if __name__ == '__main__':
    unittest.main()