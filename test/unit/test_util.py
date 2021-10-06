#!/usr/bin/env python3

import numpy as np
import os
import pytest
import torch

from botorch.exceptions import BotorchTensorDimensionError
from botorch.utils.containers import TrainingData
from scipy.optimize import Bounds

from bayesopt4ros.data_handler import DataHandler


@pytest.fixture(params=[1, 3, 10])
def test_data(request):
    """Set up a simple dataset to test the DataHandler class. The dimensionality
    of the input data is specified by the fixture parameters."""
    dim, n = request.param, 1000

    x = torch.rand(n, dim) * 10 - 5
    y = 3 + 0.5 * torch.randn(n, 1)
    return TrainingData(Xs=x, Ys=y)


def test_data_handling(test_data):
    dim = test_data.Xs.shape[1]
    bounds = Bounds(lb=-5 * np.ones((dim,)), ub=5 * np.ones((dim,)))

    # Using initilizer for setting data
    dh = DataHandler(x=test_data.Xs, y=test_data.Ys)
    x, y = dh.get_xy()
    np.testing.assert_array_equal(x, test_data.Xs)
    np.testing.assert_array_equal(y, test_data.Ys)

    d = dh.get_xy(as_dict=True)
    np.testing.assert_array_equal(d["train_inputs"], test_data.Xs)
    np.testing.assert_array_equal(d["train_targets"], test_data.Ys)

    # Using setter for setting data
    dh = DataHandler(bounds)
    np.testing.assert_equal(dh.n_data, 0)
    dh.set_xy(x=test_data.Xs, y=test_data.Ys)

    x, y = dh.get_xy()
    np.testing.assert_array_equal(x, test_data.Xs)
    np.testing.assert_array_equal(y, test_data.Ys)

    d = dh.get_xy(as_dict=True)
    np.testing.assert_array_equal(d["train_inputs"], test_data.Xs)
    np.testing.assert_array_equal(d["train_targets"], test_data.Ys)


def test_adding_data(test_data):
    dim = test_data.Xs.shape[1]

    # Single data point
    dh = DataHandler(x=test_data.Xs, y=test_data.Ys)
    x_new, y_new = torch.rand(1, dim), torch.randn(1, 1)
    dh.add_xy(x=x_new, y=y_new)
    x, y = dh.get_xy()
    np.testing.assert_array_equal(x, torch.cat((test_data.Xs, x_new)))
    np.testing.assert_array_equal(y, torch.cat((test_data.Ys, y_new)))
    np.testing.assert_equal(dh.n_data, test_data.Xs.shape[0] + 1)
    np.testing.assert_equal(len(dh), test_data.Xs.shape[0] + 1)

    # Multiple data points
    dh = DataHandler(x=test_data.Xs, y=test_data.Ys)
    x_new, y_new = torch.rand(10, dim), torch.randn(10, 1)
    dh.add_xy(x=x_new, y=y_new)
    x, y = dh.get_xy()
    np.testing.assert_array_equal(x, torch.cat((test_data.Xs, x_new)))
    np.testing.assert_array_equal(y, torch.cat((test_data.Ys, y_new)))
    np.testing.assert_equal(dh.n_data, test_data.Xs.shape[0] + 10)
    np.testing.assert_equal(len(dh), test_data.Xs.shape[0] + 10)

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
        DataHandler(x=test_data.Xs[:5], y=test_data.Ys[:6])


def test_from_single_file():
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    for dim in [1, 2]:
        data_file = os.path.join(dir, f"test_data_{dim}d_0.yaml")
        dh = DataHandler.from_file(data_file)
        x, y = dh.get_xy()
        np.testing.assert_array_equal(x, dim * torch.ones(3, dim))
        np.testing.assert_array_equal(y, dim * torch.ones(3, 1))


def test_from_multiple_files():
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    for dim in [1, 2]:
        data_files = [
            os.path.join(dir, f"test_data_{dim}d_{i}.yaml") for i in [0, 1, 2]
        ]
        dh = DataHandler.from_file(data_files)
        x, y = dh.get_xy()
        np.testing.assert_array_equal(x, dim * torch.ones(max(3 * dim, 6), dim))
        np.testing.assert_array_equal(y, dim * torch.ones(max(3 * dim, 6), 1))


def test_from_incompatible_files():
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    data_files = [
        os.path.join(dir, "test_data_1d_0.yaml"),
        os.path.join(dir, "test_data_2d_0.yaml"),
    ]

    with pytest.raises(BotorchTensorDimensionError):
        DataHandler.from_file(data_files)
