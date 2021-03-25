import argparse
import numpy as np
import rospy

from dotmap import DotMap
from scipy.optimize import Bounds


class DataHandler(object):
    """Helper class that handles all data and proper normalization thereof."""

    def __init__(
        self, bounds: Bounds, x: np.ndarray = None, y: np.ndarray = None
    ) -> None:
        # This stores the actual normalized and original data
        self.data = DotMap(
            x=np.array([]), x0=np.array([]), y=np.array([]), y0=np.array([])
        )

        # For normalization of input data
        self.bounds = bounds

        # Normalization constants for output data
        self.y_mu, self.y_std = None, None

        # When initializing with data, we assume it is not normalized
        if x is not None or y is not None:
            self.set_xy(x=x, y=y)

    def get_xy(self, norm: bool, as_dict: dict = False):
        if norm:
            x, y = self.data.x0, self.data.y0
        else:
            x, y = self.data.x, self.data.y

        if as_dict:
            return {"X": x, "Y": y}
        else:
            return (x, y)

    def set_xy(
        self,
        x: np.ndarray = None,
        x0: np.ndarray = None,
        y: np.ndarray = None,
        y0: np.ndarray = None,
    ):
        self._check_data_args(x, x0, y, y0)

        # De-/normalize inputs
        if x is not None:
            self.data.x = np.atleast_2d(x)
            self.data.x0 = self.normalize_input(self.data.x)
        else:
            self.data.x0 = np.atleast_2d(x0)
            self.data.x = self.denormalize_input(self.data.x0)

        # De-/normalize outputs
        if y is not None:
            self.data.y = np.atleast_2d(y)
            self._compute_output_normalization(self.data.y)
            self.data.y0 = self.normalize_output(self.data.y)
        else:
            self.data.y = self.denormalize_output(np.atleast_2d(y0))
            self._compute_output_normalization(self.data.y)
            self.data.y0 = self.normalize_output(self.data.y)

    def add_xy(
        self,
        x: np.ndarray = None,
        x0: np.ndarray = None,
        y: np.ndarray = None,
        y0: np.ndarray = None,
    ):
        self._check_data_args(x, x0, y, y0)

        if x is not None:
            x = np.atleast_2d(x)
            x = np.concatenate((self.data.x, x)) if self.n_data else x
        else:
            x0 = np.atleast_2d(x0)
            x0 = np.concatenate((self.data.x0, x0)) if self.n_data else x0

        if y is not None:
            y = np.atleast_2d(y)
            y = np.concatenate((self.data.y, y)) if self.n_data else y
        else:
            y0 = np.atleast_2d(y0)
            y0 = np.concatenate((self.data.y0, y0)) if self.n_data else y0

        self.set_xy(x=x, x0=x0, y=y, y0=y0)

    @property
    def n_data(self):
        return self.data.x.shape[0]

    @property
    def x_best(self):
        return self.data.x[np.argmax(self.data.y)]

    @property
    def y_best(self):
        return np.max(self.data.y)

    def __len__(self):
        return self.n_data

    def _check_data_args(
        self,
        x: np.ndarray,
        x0: np.ndarray,
        y: np.ndarray,
        y0: np.ndarray,
    ):
        if x is None and x0 is None:
            raise ValueError("Either of 'x' or 'x0' must be not None.")
        elif x is not None and x0 is not None:
            raise ValueError("Only of one 'x' or 'x0' can be not None.")
        else:
            x_tmp = x if x is not None else x0

        dim = self.bounds.lb.shape[0]
        x_tmp = np.atleast_2d(x_tmp)
        dx = x_tmp.shape[1]
        if dx != dim:
            raise ValueError(f"Input has wrong dimension: {dx}, should be {dim}")

        if y is None and y0 is None:
            raise ValueError("Either of 'y' or 'y0' must be not None.")
        elif y is not None and y0 is not None:
            raise ValueError("Only of one 'y' or 'y0' can be not None.")
        else:
            y_tmp = y if y is not None else y0

        y_tmp = np.atleast_2d(y_tmp)
        nx, ny = x_tmp.shape[0], y_tmp.shape[0]
        if nx != ny:
            raise ValueError(f"Not the same number of input/ouput data - {nx}/{ny}")

    def normalize_input(self, x):
        """Maps `x` from original domain to [0, 1]^d."""
        return (x - self.bounds.lb) / (self.bounds.ub - self.bounds.lb)

    def denormalize_input(self, x0):
        """Maps `x0` from [0, 1]^d to original domain."""
        return self.bounds.lb + (self.bounds.ub - self.bounds.lb) * x0

    def normalize_output(self, y):
        """Scale/shift `y` such that it has zero median and unit variance."""
        return (y - self.y_mu) / self.y_std

    def denormalize_output(self, y0):
        """Scale/shift `y0` such that it has original median and variance."""
        return self.y_mu + y0 * self.y_std

    def _compute_output_normalization(self, y):
        """Computes scale and shift for output normalization."""
        if self.n_data > 1:
            self.y_mu = np.median(y)
            self.y_std = np.maximum(np.std(y), 1e-3)
        else:
            self.y_mu, self.y_std = 0.0, 1.0

    def to_dict(self):
        """Writes all data to a dictionary."""
        raise NotImplementedError()


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
