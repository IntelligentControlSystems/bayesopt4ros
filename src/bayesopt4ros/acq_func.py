import abc
import numpy as np

from GPy.models import GPRegression


class AcquisitionFunction(abc.ABC):
    """Abstract base class for acquisition functions."""

    def __init__(self, gp: GPRegression) -> None:
        """Initializer for abstract acquisition function class.

        Parameters
        ----------
        gp : GPy.models.GPRegression
            Gaussian process model used for computation.
        """
        self.gp = gp

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the acquisition function.

        Parameters
        ----------
        x : numpy.ndarray
            The location at which to evaluate the acquisition function.

        Returns
        -------
        numpy.ndarray
            Acquisition function values evaluated at `x`.
        """
        return


class UpperConfidenceBound(AcquisitionFunction):
    """The upper confidence bound (UCB) acquisition function.
    
    .. math::
        \\alpha(x) = \mu(x) + \\beta \sigma(x)
    """

    def __init__(self, gp: GPRegression, beta: float = 2.0) -> None:
        """Initializer for UCB acquisition function.

        Parameters
        ----------
        gp : GPy.models.GPRegression
            Gaussian process model used for computation.
        beta : float
            Confidence multiplier governing exploration/exploitation trade-off.
        """
        super().__init__(gp=gp)
        self.beta = beta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the acquisition function.

        Parameters
        ----------
        x : numpy.ndarray
            The location at which to evaluate the acquisition function.

        Returns
        -------
        numpy.ndarray
            Acquisition function values evaluated at `x`.
        """
        mu, var = self.gp.predict(x, include_likelihood=False)
        std = np.sqrt(var)
        return mu + self.beta * std