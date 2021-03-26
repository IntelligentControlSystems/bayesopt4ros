import GPy
import numpy as np
import os
import rospy
import shutil
import sobol_seq
import yaml

from scipy.optimize import Bounds
from typing import Union

from bayesopt4ros.acq_func import UpperConfidenceBound, ExpectedImprovement
from bayesopt4ros.optim import maximize_restarts
from bayesopt4ros.util import DataHandler


class BayesianOptimization(object):
    """The Bayesian optimization class.

    Implements the actual heavy lifting that is done under the hood of  
    :class:`bayesopt_server.BayesOptServer`.

    .. note:: We assume that the objective function is to be maximized!

    .. note:: We normalize the input data to [0, 1]^d and the output data to
        zero median and unit variance. The convention within this class is that
        'x' or 'y' with a trailing zero, i.e., 'x0'/'y0' denotes that the data
        is normalized.

    .. todo:: Add flag to optionally minimize the objective instead.
    """

    def __init__(
        self,
        input_dim: int,
        max_iter: int,
        bounds: Bounds,
        acq_func: str = "UCB",
        n_init: int = 5,
        log_dir: str = None,
        config: dict = None,
    ) -> None:
        """The BayesianOptimization class initializer.

        .. note:: If a `log_dir` is specified, three different files will be 
            created: 1) evaluations file, 2) model file, 3) config file. As the
            names suggest, these store the evaluated points, the final GP model
            as well as the configuration, respectively.

        Parameters
        ----------
        input_dim : int
            Number of input dimensions for the parameters.
        max_iter : int
            Maximum number of iterations.
        bounds : scipy.optimize.Bounds
            Bounds specifying the optimization domain.
        acq_func : str  
            The acquisition function.
        n_init : int
            Number of point for initial design, i.e. Sobol.
        log_dir : str
            Directory to which the log files are stored.
        config : dict
            The configuration dictionary for the experiment.
        """
        self.input_dim = input_dim
        self.max_iter = max_iter
        self.bounds = bounds
        self.acq_func = acq_func
        self.gp = None  # GP is initialized in self._update_model()
        self.n_init = n_init
        self.x0_init = self._initial_design(n_init)
        self.x0_new = None
        self.config = config
        self.data_handler = DataHandler(bounds)

        self.log_dir = log_dir
        if self.log_dir is not None:
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
                rospy.loginfo(f"Created logging directory: {self.log_dir}")
            else:
                # TODO(lukasfro): if non-empty log_dir exists, assume that we want to continue the optimization
                rospy.logwarn(f"Logging directory already exists: {self.log_dir}")
                shutil.rmtree(self.log_dir)
                os.mkdir(self.log_dir)
            self.evaluations_file = os.path.join(self.log_dir, "evaluations.yaml")
            self.model_file = os.path.join(self.log_dir, "model")
            self.config_file = os.path.join(self.log_dir, "config.yaml")
        
        if not (bounds.lb.shape[0] == bounds.ub.shape[0] == self.input_dim):
            raise ValueError("Shape of the bounds and input dimensionality does not match.")

    @classmethod
    def from_file(cls, config_file: str):
        """Initialize a BayesianOptimization instance from a config file.

        Parameters
        ----------
        config_file : str
            The config file (full path, relative or absolute).

        Returns
        -------
        :class:`BayesianOptimization`
            An instance of the BayesianOptimization class.
        """
        # Read config from file
        try:
            with open(config_file, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

        except FileNotFoundError:
            rospy.logerr(
                f"The config file ({config_file}) you specified does not exist."
            )
            exit(1)

        # Bring bounds in correct format
        lb = np.array(config["lower_bound"])
        ub = np.array(config["upper_bound"])
        bounds = Bounds(lb=lb, ub=ub)

        # Construct class instance based on the config
        return cls(
            input_dim=config["input_dim"],
            max_iter=config["max_iter"],
            bounds=bounds,
            acq_func=config["acq_func"],
            n_init=config["n_init"],
            log_dir=config["log_dir"],
            config=config,
        )

    def next(self, y_new: float) -> np.ndarray:
        """Compute new parameters to perform an experiment with.

        The functionality of this method can generally be split into three steps:

        1) Update the model with the new data.
        2) Retrieve a new point as response of the server.
        3) Save current state to file.

        Parameters
        ----------
        y_new : float   
            The function value obtained from the most recent experiment.

        Returns
        -------
        numpy.ndarray
            The new parameters as an array.
        """
        # 1) Update the model with the new data
        if self.x0_new is not None:
            self._update_model(y_new)

        # 2) Retrieve a new point as response of the server
        if self.x0_new is None:
            # Haven't seen any data yet
            self.x0_new = self.x0_init[0]
        elif self.n_data < self.n_init:
            # Stil in the initial phase
            self.x0_new = self.x0_init[self.n_data]
        else:
            # Actually optimizing the acquisition function for new points
            self.x0_new = self._optimize_acq()

        # 3) Save current state to file
        if self.gp is not None:
            self._log_results()

        return self.data_handler.denormalize_input(self.x0_new)

    def update_last_y(self, y_last: float) -> None:
        """Updates the GP model with the last function value obtained.

        .. note:: This function is only called once from the server, right before
            shutting down the node. However, we still want to update the GP model
            with the latest data.

        Parameters
        ----------
        y_last : float
           The function value obtained from the last experiment.
        """
        self._update_model(y_last)
        self._log_results()

    @property
    def n_data(self) -> int:
        """Property for conveniently accessing number of data points."""
        return self.data_handler.n_data

    @property
    def y_best(self) -> float:
        """Get the best function value observed so far."""
        return self.data_handler.y_best

    @property
    def x_best(self) -> np.ndarray:
        """Get parameters for best function value so far."""
        return self.data_handler.x_best

    def _update_model(self, y_new: float) -> None:
        """Updates the GP with new data. Creates a model if none exists yet.

        Parameters
        ----------
        y_new : float
            The function value obtained from the last experiment.
        """
        self.data_handler.add_xy(x0=self.x0_new, y=y_new)
        x0y0 = self.data_handler.get_xy(norm=True, as_dict=True)

        if self.gp:
            self.gp.set_XY(**x0y0)
        else:
            kernel = GPy.kern.Matern52(input_dim=self.input_dim)
            self.gp = GPy.models.GPRegression(**x0y0, kernel=kernel)
            self.gp.likelihood.variance.set_prior(GPy.priors.Gamma(1.1, 0.05))
            self.gp.kern.variance.set_prior(GPy.priors.Gamma(2.0, 0.15))
            self.gp.kern.lengthscale.set_prior(GPy.priors.Gamma(3.0, 6.0))

        self.gp.optimize_restarts(num_restarts=10, verbose=False)

    def _optimize_acq(self) -> np.ndarray:
        """Optimizes the acquisition function.

        Returns
        -------
        numpy.ndarray
            Location of the acquisition function's optimum.
        """
        if self.acq_func.upper() == "UCB":
            acq_func = UpperConfidenceBound(gp=self.gp)
        elif self.acq_func.upper() == "EI":
            acq_func = ExpectedImprovement(gp=self.gp)
        else:
            raise NotImplementedError(f"{self.acq_func} is not a valid acquisition function")

        # TODO(lukasfro): Possibly expose the `n0` parameter
        x0_opt = maximize_restarts(acq_func=acq_func, n0=10)
        return x0_opt

    def _initial_design(self, n_init: int) -> np.ndarray:
        """Create initial data points from a Sobol sequence.

        Parameters
        ----------
        n_init : int
           Number of initial points.
        
        Returns
        -------
        numpy.ndarray
            Array containing the initial points.
        """
        # TODO(lukasfro): Switch from sobol_seq to Scipy.stats.qmc.Sobol when SciPy 1.7 is out
        # NOTE(lukasfro): Sobol-seq is deprecated as of very recently. The currention development
        # version of Scipy 1.7 has a new subpackage 'QuasiMonteCarlo' which implements Sobol.
        return sobol_seq.i4_sobol_generate(self.input_dim, n_init)

    def _log_results(self) -> None:
        """Log evaluations and GP model to file.

        .. note:: We do this at each iteration and overwrite the existing file in 
            case something goes wrong with either the optimization itself or on 
            the client side. We do not want to loose any valuable experimental data.
        """
        if self.log_dir:
            # Saving GP model to file
            self.gp._save_model(self.model_file, compress=False)

            # Save config to file
            yaml.dump(self.config, open(self.config_file, "w"))

            # Compute rolling best input/ouput pair
            x, y = self.data_handler.get_xy(norm=False)

            idx_best = np.array([np.argmax(y[:i+1]) for i in range(y.shape[0])])
            x_best = x[idx_best]
            y_best = y[idx_best]

            # Store all and optimal evaluation inputs/outputs to file
            data = self.data_handler.to_dict()
            data.update({"x_best": x_best, "y_best": y_best})
            data = {k: v.tolist() for k, v in data.items()}
            yaml.dump(data, open(self.evaluations_file, "w"), indent=2)
