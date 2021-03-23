import abc
import numpy as np

from typing import Tuple, Union

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

        # overwrite this if acquisition function has gradient implementation
        self.has_gradient = False  

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, jac=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Evaluate the acquisition function.

        Parameters
        ----------
        x : numpy.ndarray
            The location at which to evaluate the acquisition function.
        jac : bool, optional: False
            Specify if gradient should be returned as well (see notes).

        Returns
        -------
        numpy.ndarray
            Function values evaluated at `x` (if jac == False).
        (numpy.ndarray, numpy.ndarray)
            Function values and gradients evaluted at `x` (if jac == True).

        Notes
        -----
        Neat interfacing trick when using scipy.optimize.minimize:
        From the scipy documentation: "If jac is a Boolean and is True, fun is
        assumed to return the gradient along with the objective function. If
        False, the gradient will be estimated using ‘2-point’ finite difference
        estimation."
        """
        return


class UpperConfidenceBound(AcquisitionFunction):
    """The upper confidence bound (UCB) acquisition function.
    
    .. math::
        \\alpha(x) = \mu(x) + \\beta \sigma(x)
    """

    def __init__(self, gp: GPRegression, beta: float = 2.0) -> None:
        """Initializer for the UCB acquisition function.

        Parameters
        ----------
        gp : GPy.models.GPRegression
            Gaussian process model used for computation.
        beta : float
            Confidence multiplier governing exploration/exploitation trade-off.
        """
        super().__init__(gp=gp)
        self.beta = beta
        self.has_gradient = True

    def __call__(self, x: np.ndarray, jac=False) -> np.ndarray:
        """Evaluate the acquisition function.
        
        See base class' :func:`~AcquisitionFunction.__call__` for reference.
        """
        x = np.atleast_2d(x)
        assert self.has_gradient if jac else True
        assert x.shape[1] == self.gp.kern.input_dim

        m, v = self.gp.predict(x, include_likelihood=False)
        v = np.clip(v, 1e-10, np.inf)
        s = np.sqrt(v)
        f = m + self.beta * s

        if jac:
            dm, dv = self.gp.predictive_gradients(x)
            dm = dm[:, :, 0]
            ds = dv / (2 * s)  # applying chain rule
            g = np.squeeze(dm + self.beta * ds)  # TODO(lukasfro): why does this work?! sign error
            return f, g 
        else:
            return f
        