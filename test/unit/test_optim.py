#!/usr/bin/env python3

PKG = "test_optim"

import sys
import unittest

import numpy as np
from scipy.optimize import Bounds

from bayesopt4ros.optim import minimize_restarts


class TestOptimizer(unittest.TestCase):

    @staticmethod
    def quadratic_function(x):
        x = np.atleast_2d(x)
        return np.sum(x ** 2, axis=1)

    def test_xopt_within_bounds_1d(self):
        bounds = Bounds(lb=np.array([-1.0]), ub=np.array([1.0]))
        xopt = minimize_restarts(self.quadratic_function, bounds, n0=5)
        np.testing.assert_almost_equal(xopt, np.array([0.0]))
        
    def test_xopt_within_bounds_3d(self):
        bounds = Bounds(lb=np.array([-1.0, -1.0, -1.0]), ub=np.array([1.0, 1.0, 1.0]))
        xopt = minimize_restarts(self.quadratic_function, bounds, n0=5)
        np.testing.assert_almost_equal(xopt, np.array([0.0, 0.0, 0.0]))

    def test_xopt_on_bounds_1d(self):
        bounds = Bounds(lb=np.array([1.0]), ub=np.array([2.0]))
        xopt = minimize_restarts(self.quadratic_function, bounds, n0=5)
        np.testing.assert_almost_equal(xopt, np.array([1.0]))
        
    def test_xopt_on_bounds_3d(self):
        bounds = Bounds(lb=np.array([1.0, 1.0, 1.0]), ub=np.array([2.0, 2.0, 2.0]))
        xopt = minimize_restarts(self.quadratic_function, bounds, n0=5)
        np.testing.assert_almost_equal(xopt, np.array([1.0, 1.0, 1.0]))
