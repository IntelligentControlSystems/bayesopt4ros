import numpy as np

from typing import Callable
from scipy.optimize import Bounds, minimize


def minimize_restarts(fun: Callable, n0: int, bounds: Bounds) -> np.ndarray:
    """! Thin wrapper around scipy.optimize.minimize with random restarts.

    Note: The starting points are not chosen by a Sobol sequence by design.
          Since already the initial design relies on the Sobol sequence and
          we do not want to start the local optimizer at the data points.

    @param fun      Function to be optimized.
    @param n0       Number of restarts.
    @param bounds   Bounds for optimization.

    @return The location of the best local optimum.
    """
    dim = bounds.lb.shape[0]
    x_rand = np.random.uniform(low=bounds.lb, high=bounds.ub, size=(n0, dim))

    xopt, fopt = None, np.inf
    for x0 in x_rand:
        res = minimize(fun=fun, x0=x0, bounds=bounds)
        if res["fun"] < fopt:
            xopt, fopt = res["x"], res["fun"]
    return xopt
