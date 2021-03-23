#!/usr/bin/env python3

import numpy as np
import pytest

from scipy.optimize import Bounds

from bayesopt4ros.optim import minimize_restarts


@pytest.fixture(params=[1, 3, 10])
def centric_bounds(request):
    dim = request.param
    lb = -1.0 * np.ones((dim,))
    ub = +1.0 * np.ones((dim,))
    return Bounds(lb=lb, ub=ub)


@pytest.fixture(params=[1, 3, 10])
def offcentric_bounds(request):
    dim = request.param
    lb = +1.0 * np.ones((dim,))
    ub = +2.0 * np.ones((dim,))
    return Bounds(lb=lb, ub=ub)


def quadratic_function(x):
    x = np.atleast_2d(x)
    return np.sum(x ** 2, axis=1)


def test_xopt_within_bounds(centric_bounds):
    bounds = centric_bounds
    dim = bounds.lb.shape[0]
    xsol = np.zeros((dim,))
    xopt = minimize_restarts(quadratic_function, bounds, n0=5)
    np.testing.assert_almost_equal(xopt, xsol)


def test_xopt_on_bounds(offcentric_bounds):
    bounds = offcentric_bounds
    dim = bounds.lb.shape[0]
    xsol = np.ones((dim,))
    xopt = minimize_restarts(quadratic_function, bounds, n0=5)
    np.testing.assert_almost_equal(xopt, xsol)
