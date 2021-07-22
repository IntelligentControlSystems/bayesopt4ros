import rospy

from functools import wraps
from torch import Tensor
from typing import Callable, Optional

from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform


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