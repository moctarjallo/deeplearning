#!/usr/bin/env python
# coding: utf-8

import numpy as np
import traceback
import sys

def runtest(test,name):
    print('Running Test: %s ... ' % (name),end='')
    try:
        if test():
            print('✔ Passed!')
        else:
            print("✖ Failed!\n The output of your function does not match the expected output. Check your code and try again.")
    except Exception as e:
        print('✖ Failed!\n Your code raises an exception. The following is the traceback of the failure:')
        print(' '.join(traceback.format_tb(sys.exc_info()[2])))



def perceptron_update(x,y,w):
    """
    function w=perceptron_update(x,y,w);

    Implementation of Perceptron weights updating
    Input:
    x : input vector of d dimensions (d)
    y : corresponding label (-1 or +1)
    w : weight vector of d dimensions

    Output:
    w : weight vector after updating (d)
    """

    # your code here


    return w

def test_perceptron_update1():
    x = np.array([0,1])
    y = -1
    w = np.array([1,1])
    w1 = perceptron_update(x,y,w)
    breakpoint()
    return (w1.reshape(-1,) == np.array([1,0])).all()

def test_perceptron_update2():
    x = np.random.rand(25)
    y = 1
    w = np.zeros(25)
    w1 = perceptron_update(x,y,w)
    return np.linalg.norm(w1-x)<1e-8


def test_perceptron_update3():
    x = np.random.rand(25)
    y = -1
    w = np.zeros(25)
    w1 = perceptron_update(x,y,w)
    return np.linalg.norm(w1+x)<1e-8


runtest(test_perceptron_update1, 'test_perceptron_update1')
runtest(test_perceptron_update2, 'test_perceptron_update2')
runtest(test_perceptron_update3, 'test_perceptron_update3')