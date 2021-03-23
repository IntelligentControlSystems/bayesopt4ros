#!/usr/bin/env python3

import GPy
import numpy as np
import pytest

from scipy.optimize import check_grad

from bayesopt4ros.acq_func import UpperConfidenceBound


@pytest.fixture(params=[1, 3, 10])
def test_gp(request):
    """Set up simple Gaussian Process with random data. The dimensionality of
    the input data is specified by the fixture parameters."""
    dim = request.param
    kernel = GPy.kern.RBF(input_dim=dim)

    x = np.random.uniform(low=-1.0, high=1.0, size=(10 * dim, dim))
    y = np.sum(x ** 2, axis=1, keepdims=True)
    y += 0.05 * np.random.randn(*y.shape)

    gp = GPy.models.GPRegression(X=x, Y=y, kernel=kernel)
    gp.optimize()

    return gp


def test_ucb_grad(test_gp):
    dim = test_gp.kern.input_dim

    acq_func = UpperConfidenceBound(test_gp, beta=2.0)

    f = lambda x: acq_func(x, jac=False).squeeze()
    g = lambda x: acq_func(x, jac=True)[0].squeeze()

    for _ in range(100):
        x_eval = np.random.uniform(low=-1.0, high=1.0, size=(dim,))
        assert check_grad(f, g, x_eval)
