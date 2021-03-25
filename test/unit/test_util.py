#!/usr/bin/env python3

import numpy as np
import pytest

from dotmap import DotMap
from scipy.optimize import Bounds

from bayesopt4ros.util import DataHandler


@pytest.fixture(params=[1, 3, 10])
def test_data(request):
    """Set up a simple dataset to test the DataHandler class. The dimensionality
    of the input data is specified by the fixture parameters."""
    dim, n = request.param, 1000
    x = np.random.uniform(low=-5, high=+5, size=(n, dim))
    y = 3 + 0.5 * np.random.randn(n, 1)
    return DotMap(x=x, y=y)


def test_data_handling(test_data):
    dim = test_data.x.shape[1]
    bounds = Bounds(lb=-5 * np.ones((dim,)), ub=5 * np.ones((dim,)))
    
    # Using initilizer for setting data
    dh = DataHandler(bounds, x=test_data.x, y=test_data.y)
    x, y = dh.get_xy(norm=False)
    np.testing.assert_array_equal(x, test_data.x)
    np.testing.assert_array_equal(y, test_data.y)

    d = dh.get_xy(norm=False, as_dict=True)
    np.testing.assert_array_equal(d["X"], test_data.x)
    np.testing.assert_array_equal(d["Y"], test_data.y)

    # Using setter for setting data
    dh = DataHandler(bounds)
    np.testing.assert_equal(dh.n_data, 0)
    dh.set_xy(x=test_data.x, y=test_data.y)

    x, y = dh.get_xy(norm=False)
    np.testing.assert_array_equal(x, test_data.x)
    np.testing.assert_array_equal(y, test_data.y)

    d = dh.get_xy(norm=False, as_dict=True)
    np.testing.assert_array_equal(d["X"], test_data.x)
    np.testing.assert_array_equal(d["Y"], test_data.y)


def test_adding_data(test_data):
    dim = test_data.x.shape[1]
    bounds = Bounds(lb=-5 * np.ones((dim,)), ub=5 * np.ones((dim,)))

    # Single data point
    dh = DataHandler(bounds, x=test_data.x, y=test_data.y)
    x_new, y_new = np.random.rand(dim,), np.random.randn(1,)
    dh.add_xy(x=x_new, y=y_new)
    x, y = dh.get_xy(norm=False)
    np.testing.assert_array_equal(x, np.concatenate((test_data.x, x_new[None, :])))
    np.testing.assert_array_equal(y, np.concatenate((test_data.y, y_new[None, :])))
    np.testing.assert_equal(dh.n_data, test_data.x.shape[0] + 1)
    np.testing.assert_equal(len(dh), test_data.x.shape[0] + 1)
    
    # Multiple data points
    dh = DataHandler(bounds, x=test_data.x, y=test_data.y)
    x_new, y_new = np.random.rand(10, dim), np.random.randn(10, 1)
    dh.add_xy(x=x_new, y=y_new)
    x, y = dh.get_xy(norm=False)
    np.testing.assert_array_equal(x, np.concatenate((test_data.x, x_new)))
    np.testing.assert_array_equal(y, np.concatenate((test_data.y, y_new)))
    np.testing.assert_equal(dh.n_data, test_data.x.shape[0] + 10)
    np.testing.assert_equal(len(dh), test_data.x.shape[0] + 10)

    # Adding to empty DataHandler
    dh = DataHandler(bounds)
    x_new, y_new = np.random.rand(1, dim), np.random.randn(1, 1)
    dh.add_xy(x=x_new, y=y_new)
    x, y = dh.get_xy(norm=False)
    np.testing.assert_array_equal(x, x_new)
    np.testing.assert_array_equal(y, y_new)
    np.testing.assert_equal(dh.n_data, 1)
    np.testing.assert_equal(len(dh), 1)
    
    

def test_normalization(test_data):
    dim = test_data.x.shape[1]
    bounds = Bounds(lb=-5 * np.ones((dim,)), ub=5 * np.ones((dim,)))
    dh = DataHandler(bounds, x=test_data.x, y=test_data.y)
    x0, y0 = dh.get_xy(norm=True)

    # check input
    np.testing.assert_equal((x0 >= 0.0).all(), True)
    np.testing.assert_equal((x0 <= 1.0).all(), True)

    # check output
    np.testing.assert_almost_equal(np.std(y0), 1.0)
    np.testing.assert_almost_equal(np.median(y0), 0.0)

    dh.set_xy(x0=x0, y0=y0)
    x, y = dh.get_xy(norm=False)

    np.testing.assert_array_almost_equal(x, test_data.x)
    np.testing.assert_array_almost_equal(y, test_data.y)


def test_single_data_point(test_data):
    dim = test_data.x.shape[1]
    bounds = Bounds(lb=-5 * np.ones((dim,)), ub=5 * np.ones((dim,)))
    dh = DataHandler(bounds, x=test_data.x[0], y=test_data.y[0])
    
    np.testing.assert_equal(dh.y_mu, 0.0)
    np.testing.assert_equal(dh.y_std, 1.0)


def test_wrong_inputs(test_data):
    dim = test_data.x.shape[0]
    bounds = Bounds(lb=-5 * np.ones((dim,)), ub=5 * np.ones((dim,)))
    
    # Unequal number of inputs/outputs
    with pytest.raises(ValueError):
        dh = DataHandler(bounds, x=test_data.x[:5], y=test_data.y[:6])

    # Incorrect input dimensionaltiy
    with pytest.raises(ValueError):
        dh = DataHandler(bounds, x=test_data.x[:, :3], y=test_data.y)

    # Double assignment input
    with pytest.raises(ValueError):
        dh = DataHandler(bounds)
        dh.set_xy(x=test_data.x, x0=test_data.x, y=test_data.y)

    # Double assignment output
    with pytest.raises(ValueError):
        dh = DataHandler(bounds)
        dh.set_xy(x=test_data.x, y=test_data.y, y0=test_data.y)
