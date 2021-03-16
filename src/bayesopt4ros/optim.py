import numpy as np

from typing import Callable
from scipy.optimize import Bounds, minimize


def minimize_restarts(fun: Callable, bounds: Bounds, n0: int = 1) -> np.ndarray:
    """Thin wrapper around ``scipy.optimize.minimize`` with random restarts.

    .. note:: The starting points are not chosen by a Sobol sequence by design.
        Since already the initial design relies on the Sobol sequence and we do
        not want to start the local optimizer at the data points.

    .. todo:: Improved random starting points: sample many (>> n0) and evaluate
        acquisition function at those locations. Then randomly choose n0 points
        from the initial points where the probabilities depend on the respective
        acquisition function value.
        
    Parameters
    ----------
    fun : Callable
        Function to be minimized.
    bounds : scipy.optimize.Bounds
        Bounds for optimization.
    n0 : int
        Number of restarts.

    Returns
    -------
    numpy.ndarray
        The location of the best local optimum found.
    """

    dim = bounds.lb.shape[0]
    x_rand = np.random.uniform(low=bounds.lb, high=bounds.ub, size=(n0, dim))

    xopt, fopt = None, np.inf
    for x0 in x_rand:
        res = minimize(fun=fun, x0=x0, bounds=bounds)
        if res["fun"] < fopt:
            xopt, fopt = res["x"], res["fun"]
    return xopt
