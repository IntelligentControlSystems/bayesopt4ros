import abc
import numpy as np

from GPy.models import GPRegression


class AcquisitionFunction(abc.ABC):
    """! Abstract base class for acquisition functions. """

    def __init__(self, gp: GPRegression) -> None:
        """! Initializer for abstract acquisition function class.

        @param gp   Gaussian process model used for computation.
        """
        self.gp = gp

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """! Evaluate the acquisition function.

        @param x    The location at which to evaluate.

        @param Acquisition function evaluated at `x`.
        """
        return


class UpperConfidenceBound(AcquisitionFunction):
    """! The upper confidence bound (UCB) acquisition function. """

    def __init__(self, gp: GPRegression, beta: float = 2.0) -> None:
        """! Initializer for UCB acquisition function.

        @param gp   Gaussian process model used for computation.
        @param beta Confidence multiplier: ucb(x) = mu(x) + beta * std(x)
        """
        super().__init__(gp=gp)
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """! See documentation for abstract base class. """
        mu, var = self.gp.predict(x, include_likelihood=False)
        std = np.sqrt(var)
        return mu + self.beta * std