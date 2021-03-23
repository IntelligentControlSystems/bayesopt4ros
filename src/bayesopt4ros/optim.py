import numpy as np

from scipy.optimize import Bounds, minimize
from scipy.special import softmax
from scipy.stats import norm

from bayesopt4ros.acq_func import AcquisitionFunction


def maximize_restarts(
    acq_func: AcquisitionFunction,
    bounds: Bounds,
    n0: int = 1,
) -> np.ndarray:
    """Thin wrapper around ``scipy.optimize.minimize`` with random restarts.

    .. note:: The starting points are not chosen by a Sobol sequence by design.
        Since already the initial design relies on the Sobol sequence and we do
        not want to start the local optimizer at the data points.

    Parameters
    ----------
    acq_func : :class:`~acq_func.AcquisitionFunction`
        Acquisition function to be maximized.
    bounds : scipy.optimize.Bounds
        Bounds for optimization.
    n0 : int
        Number of restarts.

    Returns
    -------
    numpy.ndarray
        The location of the best local optimum found.
    """
    x_anchor = get_anchor_points(acq_func, bounds, n0)

    def func(x):
        # Takes care of shape mismatch between GPy and scipy.minimize
        # Recall that acquisition functions are to be maximized but scipy minimizes
        x = np.atleast_2d(x)
        if acq_func.has_gradient:
            f, g = acq_func(x, jac=True)
            return -1 * f.squeeze(), -1 * g
        else:
            return -1 * acq_func(x).squeeze()

    xopt, fopt = None, np.inf
    for x0 in x_anchor:
        res = minimize(fun=func, x0=x0, bounds=bounds, jac=acq_func.has_gradient)
        if res["fun"] < fopt:
            xopt, fopt = res["x"], res["fun"]
    return xopt


def get_anchor_points(
    acq_func: AcquisitionFunction, bounds: Bounds, n0: int = 1
) -> np.ndarray:
    """Get random starting points for optimization of the acquisition function.
    
    Parameters
    ----------
    acq_func : :class:`~acq_func.AcquisitionFunction`
        Acquisition function used for sampling.
    bounds : scipy.optimize.Bounds
        Bounds for optimization.
    n0 : int
        Number of restarts.
        
    Returns
    -------
    numpy.ndarray
        Array of anchor points.    
    """
    dim = bounds.lb.shape[0]
    x0 = np.random.uniform(low=bounds.lb, high=bounds.ub, size=(2000, dim))
    probs = softmax(acq_func(x0)).squeeze()
    idx = np.random.choice(x0.shape[0], replace=False, p=probs, size=n0)
    return x0[idx]