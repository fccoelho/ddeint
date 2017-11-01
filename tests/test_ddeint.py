__author__ = 'fccoelho'

"""
Will solve the delayed Lotka-Volterra system defined as
    
        For t < 0:
        x(t) = 1+t
        y(t) = 2-t
    
        For t >= 0:
        dx/dt =  0.5* ( 1- y(t-d) )
        dy/dt = -0.5* ( 1- x(t-d) )
    
    The delay ``d`` is a tunable parameter of the model.
"""

import unittest
import numpy as np
from ddeint import ddeint

def model(XY,t,d):
    x, y = XY(t)
    xd, yd = XY(t-d)
    return np.array([0.5*x*(1-yd), -0.5*y*(1-xd)])


class ModelTest(unittest.TestCase):
    def test_something(self):
        g = lambda t: np.array([1 + t, 2 - t])  # 'history' at t<0
        tt = np.linspace(0, 30, 20000)  # times for integration
        d = 0.5  # set parameter d
        yy = ddeint(model, g, tt, fargs=(d,))  # solve the DDE !
        self.assertEqual(yy.shape[0], 20000)


if __name__ == '__main__':
    unittest.main()
