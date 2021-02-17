from typing import List, Union
import numpy as np
import rospy
import yaml
import sobol_seq

from scipy.optimize import minimize, Bounds

from GPy.models import GPRegression

from bayesopt.acq_func import UpperConfidenceBound
from bayesopt.optim import minimize_restarts


class BayesianOptimization(object):
    """! The Bayesian optimization class.

    Implements the actual heavy lifting that is done behind the BayesOpt service.

    Note: We assume that the objective function is to be maximized!
    """

    def __init__(
        self,
        input_dim: int,
        max_iter: int,
        bounds: Bounds,
        acq_func: str,
        n_init: int,
    ) -> None:
        """! The BayesianOptimization class initializer.

        @param input_dim    Number of input dimensions for the parameters.
        @param max_iter     Maximum number of iterations.
        @param bounds       Bounds specifying the optimization domain.
        @param acq_func     Acquisition function (UCB or EI).
        @param n_init       Number of point for initial design, i.e. Sobol.
        """
        self.input_dim = input_dim
        self.max_iter = max_iter
        self.bounds = bounds
        self.acq_func = acq_func
        self.gp = None  # GP is initialized in self._update_model()
        self.n_init = n_init
        self.x_init = self._initial_design(n_init)
        self.x_new = None

        assert bounds.lb.shape[0] == bounds.ub.shape[0] == self.input_dim

    @classmethod
    def from_file(cls, settings_file: str):
        """! Initialize a BayesianOptimization instance from a settings file.

        @param settings_file    The settings file (full path, relative or absolute).

        @return An instance of the BayesianOptimization class.
        """
        # Read settings from file
        try:
            with open(settings_file, "r") as f:
                settings = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError as e:
            rospy.logerr(
                f"The settings file ({settings_file}) you specified does not exist."
            )
            exit(1)

        # Bring bounds in correct format (2 x input_dim)
        lb = np.array(settings["lower_bound"])
        ub = np.array(settings["upper_bound"])
        bounds = Bounds(lb=lb, ub=ub)

        # Construct class instance based on the settings
        return cls(
            input_dim=settings["input_dim"],
            max_iter=settings["max_iter"],
            bounds=bounds,
            acq_func=settings["acq_func"],
            n_init=settings["n_init"],
        )

    def next(self, y_new: float) -> np.ndarray:
        """! Compute new parameters to perform an experiment with.

        The functionality of this method can generally be split into three parts:

        1) Update the model with the new data.
        2) Retrieve a new point as response of the service.
        3) Save current state to file.

        @param y_new    The function value obtained from the last experiment.

        @return The new parameters as an array.
        """
        # 1) Update the model with the new data
        if self.x_new is not None:
            self._update_model(self.x_new, y_new)

        # 2) Retrieve a new point as response of the service
        if self.x_new is None:
            # Haven't seen any data yet
            self.x_new = self.x_init[0]
        elif self.n_data < self.n_init:
            # Stil in the initial phase
            self.x_new = self.x_init[self.n_data]
        else:
            # Actually optimizing the acquisition function for new points
            self.x_new = self._optimize_acq()

        # 3) Save current state to file
        # TODO(lukasfro): Store current model to file.

        return self.x_new

    @property
    def n_data(self) -> int:
        """! Property for conveniently accessing number of data points.

        @return The number of data points in the GP model.
        """
        return self.gp.X.shape[0]

    def _update_model(self, x_new: np.ndarray, y_new: Union[float, np.ndarray]) -> None:
        """! Updates the GP with new data. Creates a model if none exists yet.

        @param x_new    The parameter from the last experiment.
        @param y_new    The function value obtained from the last experient.
        """
        x_new, y_new = np.atleast_2d(x_new), np.atleast_2d(y_new)
        assert x_new.ndim == 2 and y_new.ndim == 2
        assert x_new.shape[0] == y_new.shape[0]

        if self.gp:
            X = np.concatenate((self.gp.X, x_new))
            Y = np.concatenate((self.gp.Y, y_new))
            self.gp.set_XY(X=X, Y=Y)
        else:
            # TODO(lukasfro): Choose proper kernel with hyperparameters
            self.gp = GPRegression(X=x_new, Y=y_new)
        self.gp.optimize_restarts(num_restarts=10, verbose=False)

    def _optimize_acq(self) -> np.ndarray:
        """! Optimizes the acquisition function.

        @return Location of the acquisition function's optimum.
        """
        if self.acq_func.upper() == "UCB":
            acq_func = UpperConfidenceBound(gp=self.gp, beta=2.0)
        else:
            raise NotImplementedError("Only UCB is currently implemented.")

        # Takes care of dimensionality mismatch between GPy and scipy.minimize
        # Recall that acquisition functions are to be maximized but scipy minimizes
        def fun(x):
            x = np.atleast_2d(x)
            return -1 * acq_func(x).squeeze()

        # TODO(lukasfro): Possibly expose the `n0` parameter
        xopt = minimize_restarts(fun=fun, n0=10, bounds=self.bounds)

        return xopt

    def _initial_design(self, n_init: int) -> np.ndarray:
        """! Create initial data points from a Sobol sequence.

        @param n_init   Number of initial points.

        @return Array containing the initial points.
        """
        # TODO(lukasfro): Switch from sobol_seq to Scipy.stats.qmc.Sobol when SciPy 1.7 is out
        # NOTE(lukasfro): Sobol-seq is deprecated as of very recently. The currention development
        # version of Scipy 1.7 has a new subpackage 'QuasiMonteCarlo' which implements Sobol.
        return sobol_seq.i4_sobol_generate(self.input_dim, n_init)
