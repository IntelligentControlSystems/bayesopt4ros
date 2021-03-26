import numpy as np
import os
import rospy
import shutil
import torch
import yaml

from torch import Tensor

from bayesopt4ros.util import DataHandler

from botorch.acquisition import (
    UpperConfidenceBound,
    ExpectedImprovement,
    NoisyExpectedImprovement,
)
from botorch.fit import fit_gpytorch_scipy
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from gpytorch.mlls import ExactMarginalLogLikelihood


class BayesianOptimization(object):
    """The Bayesian optimization class.

    Implements the actual heavy lifting that is done under the hood of
    :class:`bayesopt_server.BayesOptServer`.

    .. note:: We assume that the objective function is to be maximized!

    .. todo:: Add flag to optionally minimize the objective instead.
    """

    def __init__(
        self,
        input_dim: int,
        max_iter: int,
        bounds: Tensor,
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
        bounds : torch.Tensor
            A [2, dim]-dim tensor specifying the optimization domain.
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
        self.gp = None  # GP is initialized when first data arrives
        self.n_init = n_init
        self.x_init = self._initial_design(n_init)
        self.x_new = None
        self.config = config
        self.data_handler = DataHandler()

        self.log_dir = log_dir
        # TODO(lukasfro): make a separate function for this
        if self.log_dir is not None:
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
                rospy.loginfo(f"Created logging directory: {self.log_dir}")
            else:
                # TODO(lukasfro): if non-empty log_dir exists, assume that we want to continue the optimization
                rospy.logwarn(f"Logging directory already exists: {self.log_dir}")
                shutil.rmtree(self.log_dir)
                os.mkdir(self.log_dir)

        if not (bounds.shape[1] == self.input_dim):
            raise ValueError("Bounds do not match input dimensionality.")

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
        lb = torch.tensor(config["lower_bound"])
        ub = torch.tensor(config["upper_bound"])
        bounds = torch.stack((lb, ub))

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

    def next(self, y_new: float) -> Tensor:
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
        torch.Tensor
            The new parameters as an array.
        """
        # 1) Update the model with the new data
        self._update_model(y_new)

        # 2) Retrieve a new point as response of the server
        self.x_new = self._get_next_x()

        # 3) Save current state to file
        self._log_results()

        return self.x_new.squeeze(0)

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

    def _get_next_x(self):
        if self.x_new is None:
            # Haven't seen any data yet
            x_new = self.x_init[[0]]
        elif self.n_data < self.n_init:
            # Stil in the initial phase
            x_new = self.x_init[[self.n_data]]
        else:
            # Actually optimizing the acquisition function for new points
            x_new = self._optimize_acq()
        return x_new

    @property
    def n_data(self) -> int:
        """Property for conveniently accessing number of data points."""
        return self.data_handler.n_data

    @property
    def y_best(self) -> float:
        """Get the best function value observed so far."""
        return self.data_handler.y_best

    @property
    def x_best(self) -> Tensor:
        """Get parameters for best function value so far."""
        return self.data_handler.x_best

    def _update_model(self, y_new: float) -> None:
        """Updates the GP with new data. Creates a model if none exists yet.

        Parameters
        ----------
        y_new : float
            The function value obtained from the last experiment.
        """
        if self.x_new is None:
            # The very first function value we obtain from the client is just to
            # trigger the server. At that point, there is no new input point,
            # hence, no need to need to update the model.
            return
        self.data_handler.add_xy(x=self.x_new, y=torch.tensor([[y_new]]))

        if self.n_data >= self.n_init:
            # Only create model once we are done with the initial design phase
            if self.gp:
                x, y = self.data_handler.get_xy()
                self.gp.set_train_data(x, y.squeeze(-1), strict=False)
            else:
                self.gp = self._initialize_model(*self.data_handler.get_xy())
            self._optimize_model()

    @staticmethod
    def _initialize_model(x, y) -> GPyTorchModel:
        # Note: the default values from BoTorch are quite good
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            outcome_transform=Standardize(m=1),  # zero mean, unit variance
            input_transform=Normalize(d=x.shape[1]),  # unit cube
        )
        return gp

    def _optimize_model(self) -> None:
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        mll.train()
        fit_gpytorch_scipy(mll)
        mll.eval()

    def _optimize_acq(self) -> Tensor:
        """Optimizes the acquisition function.

        Returns
        -------
        torch.Tensor
            Location of the acquisition function's optimum.
        """
        if self.acq_func.upper() == "UCB":
            acq_func = UpperConfidenceBound(model=self.gp, beta=4.0)
        elif self.acq_func.upper() == "EI":
            best_f = self.data_handler.y_best  # note that EI assumes noiseless
            acq_func = ExpectedImprovement(model=self.gp, best_f=best_f)
        elif self.acq_func.upper() == "NEI":
            raise NotImplementedError("Coming soon...")
        else:
            raise NotImplementedError(
                f"{self.acq_func} is not a valid acquisition function"
            )

        x_opt, _ = optimize_acqf(
            acq_func, self.bounds, q=1, num_restarts=10, raw_samples=2000, sequential=True
        )
        return x_opt

    def _initial_design(self, n_init: int) -> Tensor:
        """Create initial data points from a Sobol sequence.

        Parameters
        ----------
        n_init : int
           Number of initial points.

        Returns
        -------
        torch.Tensor
            Array containing the initial points.
        """
        sobol_eng = torch.quasirandom.SobolEngine(dimension=self.input_dim)
        sobol_eng.fast_forward(n=1)  # first point is origin, boring...
        x0_init = sobol_eng.draw(n_init)  # points are in [0, 1]^d
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * x0_init

    def _log_results(self) -> None:
        """Log evaluations and GP model to file.

        .. note:: We do this at each iteration and overwrite the existing file in
            case something goes wrong with either the optimization itself or on
            the client side. We do not want to loose any valuable experimental data.
        """
        if self.log_dir and self.gp is not None:
            
            # Saving GP model to file
            self.model_file = os.path.join(self.log_dir, "model_state.pth")
            torch.save(self.gp.state_dict(), self.model_file)

            # Save config to file
            self.config_file = os.path.join(self.log_dir, "config.yaml")
            yaml.dump(self.config, open(self.config_file, "w"))

            # Compute rolling best input/ouput pair
            x, y = self.data_handler.get_xy()

            idx_best = torch.tensor([torch.argmax(y[: i + 1]) for i in range(y.shape[0])])
            x_best = x[idx_best]
            y_best = y[idx_best]

            # Store all and optimal evaluation inputs/outputs to file
            data = self.data_handler.get_xy(as_dict=True)
            data.update({"x_best": x_best, "y_best": y_best})
            data = {k: v.tolist() for k, v in data.items()}
            self.evaluations_file = os.path.join(self.log_dir, "evaluations.yaml")
            yaml.dump(data, open(self.evaluations_file, "w"), indent=2)
