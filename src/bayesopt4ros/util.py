from __future__ import annotations

import rospy
import torch
import yaml

from functools import wraps
from torch import Tensor
from typing import Union, Callable, List, Optional

from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.model import Model
from botorch.utils.containers import TrainingData
from botorch.utils.transforms import t_batch_mode_transform


def count_requests(func: Callable) -> Callable:
    """Decorator that keeps track of number of requests.

    Parameters
    ----------
    func : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.request_count += 1
        ret_val = func(self, *args, **kwargs)
        return ret_val

    return wrapper


class PosteriorMean(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        objective: Optional[ScalarizedObjective] = None,
        maximize: bool = True,
    ) -> None:
        super().__init__(model, objective=objective)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior mean on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Posterior Mean values at the given design
            points `X`.
        """
        posterior = self._get_posterior(X=X)
        mean = posterior.mean.view(X.shape[:-2])
        if self.maximize:
            return mean
        else:
            return -mean


class DataHandler(object):
    # TODO(lukasfro): documentation for DataHandler
    # TODO(lukasfro): add `maximize` flag and just multiply y by -1 instead of keeping track of min/max
    """Helper class that handles all data for BayesOpt."""

    def __init__(self, x: Tensor = None, y: Tensor = None) -> None:
        self.set_xy(x=x, y=y)

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
    def x_min(self):
        return self.data.Xs[torch.argmin(self.data.Ys)]

    @property
    def y_min(self):
        return torch.min(self.data.Ys)

    @property
    def x_max(self):
        return self.data.Xs[torch.argmax(self.data.Ys)]

    @property
    def y_max(self):
        return torch.max(self.data.Ys)

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


def iter_to_string(it, format_spec, separator=", "):
    """Represents an iterable (list, tuple, etc.) as a formatted string.

    Parameters
    ----------
    it : Iterable
        An iterable with numeric elements.
    format_spec : str
        Format specifier according to https://docs.python.org/3/library/string.html#format-specification-mini-language
    separator : str
        String between items of the iterator.

    Returns
    -------
    str
        The iterable as formatted string.
    """
    return separator.join([f"{format(elem, format_spec)}" for elem in it])


def create_log_dir(log_dir):
    """Creates a new logging sub-directory with current date and time.

    Parameters
    ----------
    log_dir : str
        Path to the root logging directory.

    Returns
    -------
    str
        The final sub-directory path.
    """
    import os
    import time

    log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    rospy.loginfo(f"Logging directory: {log_dir}")

    return log_dir