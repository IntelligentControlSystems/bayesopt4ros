from __future__ import annotations

import rospy
import torch
import yaml

from torch import Tensor
from typing import Union, List

from botorch.utils.containers import TrainingData
from botorch.exceptions.errors import BotorchTensorDimensionError


class DataHandler(object):
    # TODO(lukasfro): documentation for DataHandler
    """Helper class that handles all data for BayesOpt."""

    def __init__(self, x: Tensor = None, y: Tensor = None, maximize: bool = True) -> None:
        self.set_xy(x=x, y=y)
        self.maximize = maximize

    @classmethod
    def from_file(cls, file: Union[str, List[str]]) -> DataHandler:
        """Creates a DataHandler instance with input/target values from the
        specified file.

        Returns an empty DataHandler object if file could not be found.
        """
        files = [file] if isinstance(file, str) else file
        x, y = [], []

        for file in files:
            try:
                with open(file, "r") as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)
                x.append(torch.tensor(data["train_inputs"]))
                y.append(torch.tensor(data["train_targets"]))
            except FileNotFoundError:
                rospy.logwarn(f"The evaluations file '{file}' could not be found.")

        if x and y:
            if not len(set([xi.shape[1] for xi in x])) == 1:  # check for correct dimension
                message = "Evaluation points seem to have different dimensions."
                raise BotorchTensorDimensionError(message)
            x = torch.cat(x)
            y = torch.cat(y)
            return cls(x=x, y=y)
        else:
            return cls()

    def get_xy(self, as_dict: dict = False):
        if as_dict:
            return {"train_inputs": self.data.Xs, "train_targets": self.data.Ys}
        else:
            return (self.data.Xs, self.data.Ys)

    def set_xy(self, x: Tensor = None, y: Union[float, Tensor] = None):
        if x is None or y is None:
            self.data = TrainingData(Xs=torch.tensor([]), Ys=torch.tensor([]))
        else:
            if not isinstance(y, Tensor):
                y = torch.tensor([[y]])
            self._validate_data_args(x, y)
            self.data = TrainingData(Xs=x, Ys=y)

    def add_xy(self, x: Tensor = None, y: Union[float, Tensor] = None):
        if not isinstance(y, Tensor):
            y = torch.tensor([[y]])
        x = torch.atleast_2d(x)
        self._validate_data_args(x, y)
        x = torch.cat((self.data.Xs, x)) if self.n_data else x
        y = torch.cat((self.data.Ys, y)) if self.n_data else y
        self.set_xy(x=x, y=y)

    @property
    def n_data(self):
        return self.data.Xs.shape[0]

    @property
    def x_best(self):
        if self.maximize:
            return self.data.Xs[torch.argmax(self.data.Ys)]
        else:
            return self.data.Xs[torch.argmin(self.data.Ys)]

    @property
    def x_best_accumulate(self):
        return self.data.Xs[self.idx_best_accumulate]

    @property
    def y_best(self):
        if self.maximize:
            return torch.max(self.data.Ys)
        else:
            return torch.min(self.data.Ys)

    @property
    def y_best_accumulate(self):
        return self.data.Ys[self.idx_best_accumulate]

    @property
    def idx_best_accumulate(self):
        argminmax = torch.argmax if self.maximize else torch.argmin
        return [argminmax(self.data.Ys[: i + 1]).item() for i in range(self.n_data)]

    def __len__(self):
        return self.n_data

    @staticmethod
    def _validate_data_args(x: Tensor, y: Tensor):
        if x.dim() != 2:
            message = f"Input dimension is assumed 2-dim. not {x.ndim}-dim."
            raise BotorchTensorDimensionError(message)
        if y.dim() != 2:
            message = f"Output dimension is assumed 2-dim. not {x.ndim}-dim."
            raise BotorchTensorDimensionError(message)
        if y.shape[1] != 1:
            message = "We only support 1-dimensional outputs for the moment."
            raise BotorchTensorDimensionError(message)
        if x.shape[0] != y.shape[0]:
            message = "Not the number of input/ouput data."
            raise BotorchTensorDimensionError(message)
