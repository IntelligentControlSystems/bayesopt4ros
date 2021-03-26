#!/usr/bin/env python3

import numpy as np
import pytest
import torch

from botorch.exceptions import BotorchTensorDimensionError
from botorch.utils.containers import TrainingData
from scipy.optimize import Bounds

from bayesopt4ros.util import DataHandler


@pytest.fixture(params=[1, 3, 10])
def test_data(request):
    """Set up a simple dataset to test the DataHandler class. The dimensionality
    of the input data is specified by the fixture parameters."""
    dim, n = request.param, 1000

    x = torch.rand(n, dim) * 10 - 5
    y = 3 + 0.5 * torch.randn(n, 1)
    return TrainingData(X=x, Y=y)


def test_data_handling(test_data):
    dim = test_data.X.shape[1]
    bounds = Bounds(lb=-5 * np.ones((dim,)), ub=5 * np.ones((dim,)))

    # Using initilizer for setting data
    dh = DataHandler(x=test_data.X, y=test_data.Y)
    x, y = dh.get_xy()
    np.testing.assert_array_equal(x, test_data.X)
    np.testing.assert_array_equal(y, test_data.Y)

    d = dh.get_xy(as_dict=True)
    np.testing.assert_array_equal(d["train_inputs"], test_data.X)
    np.testing.assert_array_equal(d["train_targets"], test_data.Y)

    # Using setter for setting data
    dh = DataHandler(bounds)
    np.testing.assert_equal(dh.n_data, 0)
    dh.set_xy(x=test_data.X, y=test_data.Y)

    x, y = dh.get_xy()
    np.testing.assert_array_equal(x, test_data.X)
    np.testing.assert_array_equal(y, test_data.Y)

    d = dh.get_xy(as_dict=True)
    np.testing.assert_array_equal(d["train_inputs"], test_data.X)
    np.testing.assert_array_equal(d["train_targets"], test_data.Y)


def test_adding_data(test_data):
    dim = test_data.X.shape[1]

    # Single data point
    dh = DataHandler(x=test_data.X, y=test_data.Y)
    x_new, y_new = torch.rand(1, dim), torch.randn(1, 1)
    dh.add_xy(x=x_new, y=y_new)
    x, y = dh.get_xy()
    np.testing.assert_array_equal(x, torch.cat((test_data.X, x_new)))
    np.testing.assert_array_equal(y, torch.cat((test_data.Y, y_new)))
    np.testing.assert_equal(dh.n_data, test_data.X.shape[0] + 1)
    np.testing.assert_equal(len(dh), test_data.X.shape[0] + 1)

    # Multiple data points
    dh = DataHandler(x=test_data.X, y=test_data.Y)
    x_new, y_new = torch.rand(10, dim), torch.randn(10, 1)
    dh.add_xy(x=x_new, y=y_new)
    x, y = dh.get_xy()
    np.testing.assert_array_equal(x, torch.cat((test_data.X, x_new)))
    np.testing.assert_array_equal(y, torch.cat((test_data.Y, y_new)))
    np.testing.assert_equal(dh.n_data, test_data.X.shape[0] + 10)
    np.testing.assert_equal(len(dh), test_data.X.shape[0] + 10)

    # Adding to empty DataHandler
    dh = DataHandler()
    x_new, y_new = torch.rand(1, dim), torch.randn(1, 1)
    dh.add_xy(x=x_new, y=y_new)
    x, y = dh.get_xy()
    np.testing.assert_array_equal(x, x_new)
    np.testing.assert_array_equal(y, y_new)
    np.testing.assert_equal(dh.n_data, 1)
    np.testing.assert_equal(len(dh), 1)


def test_wrong_inputs(test_data):

    # Unequal number of inputs/outputs
    with pytest.raises(BotorchTensorDimensionError):
        dh = DataHandler(x=test_data.X[:5], y=test_data.Y[:6])
