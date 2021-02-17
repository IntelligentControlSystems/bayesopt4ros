from typing import List, Union
import numpy as np
import rospy
import yaml

from GPy.models import GPRegression


class BayesianOptimization(object):
    """! The Bayesian optimization class.

    Implements the actual heavy lifting that is done behind the BayesOpt service.
    """

    def __init__(
        self,
        input_dim: int,
        max_iter: int,
        bounds: np.ndarray,
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
        self.gp = None  # GP is initialized in self.next()
        self.x, self.y = None, None
        self.n_init = n_init
        self.x_init = self._initial_design(n_init)
        self.x_new = None

        assert bounds.shape == (2, self.input_dim)

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
        bounds = np.stack((lb, ub))

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

        @param y_new    The function value obtained from the last experiment.

        @return The new parameters as an array.
        """
        # 1) Update GP model with new data (self.x_new from previous call, y_new)
        # 2) Fit GP model
        # 3) Optimize acquisition function
        # 4) Store data
        if self.x_new is not None:
            self._update_model(self.x_new, y_new)

        if self.x_new is None:
            # Haven't seen any data yet
            self.x_new = self.x_init[0]
            rospy.loginfo("First data point was selected")
        elif self.n_data < self.n_init:
            # Stil in the initial phase
            self.x_new = self.x_init[self.n_data]
            rospy.loginfo("Data point from initial data was selected")
        else:
            # Actually optimizing the acquisition function for new points
            self.x_new = np.random.rand(self.input_dim)
            rospy.loginfo("Data point was selected from acq func")

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

    def _initial_design(self, n_init: int) -> np.ndarray:
        """! Create initial data points from a Sobol sequence.

        @param n_init   Number of initial points.

        @return Array containing the initial points.
        """
        rospy.logwarn("Initial design is still random, no Sobol yet.")
        return np.random.rand(n_init, self.input_dim)