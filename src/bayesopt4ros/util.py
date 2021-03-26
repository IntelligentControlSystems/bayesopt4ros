import argparse
import rospy
import torch

from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.utils.containers import TrainingData
from torch import Tensor
from typing import Union


class DataHandler(object):
    """Helper class that handles all data for BayesOpt."""

    def __init__(self, x: Tensor = None, y: Tensor = None) -> None:
        # This stores the actual normalized and original data
        if x is not None and y is not None:
            self.set_xy(x=x, y=y)
        else:
            self.data = TrainingData(X=torch.tensor([]), Y=torch.tensor([]))

    def get_xy(self, as_dict: dict = False):
        if as_dict:
            return {"train_inputs": self.data.X, "train_targets": self.data.Y}
        else:
            return (self.data.X, self.data.Y)

    def set_xy(self, x: Tensor = None, y: Union[float, Tensor] = None):
        if not isinstance(y, Tensor):
            y = torch.tensor([[y]])
        self._validate_data_args(x, y)
        self.data = TrainingData(X=x, Y=y)
        
    def add_xy(self, x: Tensor = None, y: Union[float, Tensor] = None):
        if not isinstance(y, Tensor):
            y = torch.tensor([[y]])
        self._validate_data_args(x, y)
        x = torch.cat((self.data.X, x)) if self.n_data else x
        y = torch.cat((self.data.Y, y)) if self.n_data else y
        self.set_xy(x=x, y=y)

    @property
    def n_data(self):
        return self.data.X.shape[0]

    @property
    def x_best(self):
        return self.data.X[torch.argmax(self.data.Y)]

    @property
    def y_best(self):
        return torch.max(self.data.Y)

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


def server_argparser():
    """Sets up the argument parser used for the BayesOpt server."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--config_file",
        help="File containing the configuration for the Bayesian Optimization server",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--log_level",
        help="Specify the verbosity of terminal output",
        type=int,
        choices=[rospy.DEBUG, rospy.INFO, rospy.WARN],
        default=rospy.INFO,
    )

    parser.add_argument(
        "--silent",
        help="Prevents printing status/updates for the server",
        action="store_true",
    )

    return parser
