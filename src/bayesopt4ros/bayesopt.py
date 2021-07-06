from __future__ import annotations

import os
from botorch.acquisition.analytic import PosteriorMean
import rospy
import shutil
import time
import torch
import yaml

from torch import Tensor
from typing import List, Tuple

from botorch.acquisition import (
    AcquisitionFunction,
    UpperConfidenceBound,
    ExpectedImprovement,
    NoisyExpectedImprovement,
)
from botorch.fit import fit_gpytorch_scipy
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf as optimize_acqf_botorch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from gpytorch.mlls import ExactMarginalLogLikelihood

from bayesopt4ros import util
from bayesopt4ros.util import DataHandler
from bayesopt4ros.msg import BayesOptAction
from bayesopt4ros.util import NegativePosteriorMean


class BayesianOptimization(object):
    """The Bayesian optimization class.

    Implements the actual heavy lifting that is done under the hood of
    :class:`bayesopt_server.BayesOptServer`.

    """

    def __init__(
        self,
        input_dim: int,
        max_iter: int,
        bounds: Tensor,
        acq_func: str = "UCB",
        n_init: int = 5,
        log_dir: str = None,
        load_dir: str = None,
        config: dict = None,
        maximize: bool = True,
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
            A [2, input_dim] shaped tensor specifying the optimization domain.
        acq_func : str
            The acquisition function specifier.
        n_init : int
            Number of point for initial design, i.e. Sobol.
        log_dir : str
            Directory to which the log files are stored.
        load_dir : str or list of str
            Directory/directories from which initial data points are loaded.
        config : dict
            The configuration dictionary for the experiment.
        maximize : bool
            If True, consider the problem a maximization problem.
        """
        self.input_dim = input_dim
        self.max_iter = max_iter
        self.bounds = bounds
        self.acq_func = acq_func
        self.n_init = n_init
        self.x_init = self._initial_design(n_init)
        self.x_new = None
        self.config = config
        self.maximize = maximize
        self.data_handler = DataHandler()
        self.gp = None  # GP is initialized when first data arrives
        self.x_opt = torch.empty(0, input_dim)
        self.y_opt = torch.empty(0, 1)

        if load_dir is not None:
            self.data_handler, self.gp = self._load_prev_bayesopt(load_dir)

        if log_dir is not None:
            self.log_dir = util.create_log_dir(log_dir)

        assert bounds.shape[1] == self.input_dim

    @classmethod
    def from_file(cls, config_file: str) -> BayesianOptimization:
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
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

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
            load_dir=config.get("load_dir"),
            maximize=config["maximize"],
            config=config,
        )

    def next(self, goal: BayesOptAction) -> Tensor:
        """Compute new parameters to perform an experiment with.

        The functionality of this method can generally be split into three steps:

        1) Update the model with the new data.
        2) Retrieve a new point as response of the server.
        3) Save current state to file.

        Parameters
        ----------
        goal : BayesOptAction
            The goal sent from the client for the most recent experiment.

        Returns
        -------
        torch.Tensor
            The new parameters as an array.
        """
        # 1) Update the model with the new data
        self._update_model(goal)

        # 2) Retrieve a new point as response of the server
        self.x_new = self._get_next_x()

        # 3) Save current state to file
        self._log_results()

        return self.x_new

    def update_last_goal(self, goal: float) -> None:
        """Updates the GP model with the last function value obtained.

        .. note:: This function is only called once from the server, right before
            shutting down the node. However, we still want to update the GP model
            with the latest data.

        Parameters
        ----------
        goal : BayesOptAction
            The goal sent from the client for the last recent experiment.
        """
        self._update_model(goal)
        self._log_results()

    def get_optimal_parameters(self) -> Tuple[torch.Tensor, float]:
        """Get the optimal parameters with corresponding expected value."""
        return self._optimize_posterior_mean()

    def get_best_observation(self) -> Tuple[torch.Tensor, float]:
        """Get the best parameters and corresponding observed value."""
        return self.x_best, self.y_best

    @property
    def constant_config_parameters(self) -> List[str]:
        """These parameters need to be the same when loading previous runs. For
        all other settings, the user might have a reasonable explanation to
        change it inbetween experiments/runs. E.g., maximum number of iterations
        or bounds.

        See Also
        --------
        _check_config
        """
        return ["input_dim", "maximize"]

    @property
    def n_data(self) -> int:
        """Property for conveniently accessing number of data points."""
        return self.data_handler.n_data

    @property
    def y_best(self) -> float:
        """Get the best function value observed so far."""
        return self.data_handler.y_max if self.maximize else self.data_handler.y_min

    @property
    def x_best(self) -> Tensor:
        """Get parameters for best observed function value so far."""
        return self.data_handler.x_max if self.maximize else self.data_handler.x_min

    def _get_next_x(self):
        if self.n_data < self.n_init:  # We are in the initialization phase
            x_new = self.x_init[self.n_data]
        else:  # Actually optimizing the acquisition function for new points
            x_new = self._optimize_acqf()
        return x_new

    def _check_config(self, load_dirs):
        """Make sure that all relevant parameters in the configs match."""
        for load_dir in load_dirs:
            with open(os.path.join(load_dir, "config.yaml")) as f:
                load_config = yaml.load(f, Loader=yaml.FullLoader)

            for p in self.constant_config_parameters:
                try:
                    assert load_config[p] == self.__getattribute__(p)
                except AssertionError:
                    rospy.logerr(f"Your configuration does not match with {load_dir}")

    def _load_prev_bayesopt(self, load_dirs):
        # We can load multiple previous runs
        load_dirs = [load_dirs] if isinstance(load_dirs, str) else load_dirs

        # Configurations need to be compatible with the current one
        self._check_config(load_dirs)

        # Create model with the previous runs' data
        data_files = [os.path.join(load_dir, "evaluations.yaml") for load_dir in load_dirs]
        self.data_handler = DataHandler.from_file(data_files)
        self.gp = self._initialize_model(*self.data_handler.get_xy())

        return self.data_handler, self.gp

    def _update_model(self, goal) -> None:
        """Updates the GP with new data. Creates a model if none exists yet.

        Parameters
        ----------
        goal : BayesOptAction
            The goal sent from the client for the most recent experiment.
        """
        if self.x_new is None:
            # The very first function value we obtain from the client is just to
            # trigger the server. At that point, there is no new input point,
            # hence, no need to need to update the model.
            return
        self.data_handler.add_xy(x=self.x_new, y=goal.y_new)

        # Either create or update the GP model with the new data
        if not self.gp:
            self.gp = self._initialize_model(*self.data_handler.get_xy())
        else:
            x, y = self.data_handler.get_xy()
            self.gp.set_train_data(x, y.squeeze(-1), strict=False)

        self._optimize_model()

    def _initialize_model(self, x, y) -> GPyTorchModel:
        # Note: the default values from BoTorch are quite good
        unit_cube = torch.stack((torch.zeros(self.input_dim), torch.ones(self.input_dim)))
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            outcome_transform=Standardize(m=1),  # zero mean, unit variance
            input_transform=Normalize(d=self.input_dim, bounds=unit_cube),
        )
        return gp

    def _optimize_model(self) -> None:
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        mll.train()
        fit_gpytorch_scipy(mll)
        mll.eval()

    def _initialize_acqf(self) -> AcquisitionFunction:
        """Initialize the acquisition function of choice.

        Returns
        -------
        AcquisitionFunction
            An acquisition function based on BoTorch's base class.
        """
        if self.acq_func.upper() == "UCB":
            acq_func = UpperConfidenceBound(model=self.gp, beta=4.0, maximize=self.maximize)
        elif self.acq_func.upper() == "EI":
            best_f = self.y_best  # note that EI assumes noiseless
            acq_func = ExpectedImprovement(model=self.gp, best_f=best_f, maximize=self.maximize)
        elif self.acq_func.upper() == "NEI":
            # TODO(lukasfro): implement usage for Noisy EI
            raise NotImplementedError("Coming soon...")
        else:
            raise NotImplementedError(f"{self.acq_func} is not a valid acquisition function")
        return acq_func

    def _optimize_acqf(self) -> Tensor:
        """Optimizes the acquisition function.

        Returns
        -------
        torch.Tensor
            Location of the acquisition function's optimum.
        """
        acq_func = self._initialize_acqf()
        x_opt, f_opt = optimize_acqf_botorch(
            acq_func,
            self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=2000,
            sequential=True,
        )

        self._debug_acqf_visualize(acq_func, x_opt, f_opt)

        x_opt = x_opt.squeeze(0)  # gets rid of superfluous dimension due to q=1
        return x_opt

    def _debug_acqf_visualize(self, acq_func, x_opt, f_opt):
        """Visualize the acquisition function for debugging purposes."""
        import matplotlib.pyplot as plt

        if not self.input_dim == 2:
            return

        # The plotting ranges
        lb, ub = self.bounds[0], self.bounds[1]
        x1 = torch.linspace(lb[0], ub[0], 100)
        x2 = torch.linspace(lb[1], ub[1], 100)
        x1, x2 = torch.meshgrid(x1, x2)
        xs = torch.stack((x1.flatten(), x2.flatten())).T

        # Evaluate GP and acquisition function
        gpm = self.gp.posterior(xs).mean.squeeze().detach().view(100, 100)
        acqf = acq_func(xs.unsqueeze(1)).squeeze().detach().view(100, 100)

        x_eval = self.data_handler.get_xy()[0]

        fig, axes = plt.subplots(nrows=1, ncols=2)
        c0 = axes[0].contourf(x1, x2, gpm, levels=50)
        axes[0].plot(x_eval[:, 0], x_eval[:, 1], "ko")
        axes[0].axis("equal")
        c1 = axes[1].contourf(x1, x2, acqf, levels=50)
        axes[1].plot(x_opt[0, 0], x_opt[0, 1], "C3o")
        axes[1].axis("equal")

        # fig.colorbar(c0, axes[0])
        # fig.colorbar(c1, axes[1])

        plt.tight_layout()

        file_name = os.path.join(self.log_dir, f"acqf_visualize_{x_eval.shape[0]}.pdf")
        rospy.logwarn(f"Saving debug visualization to: {file_name}")
        plt.savefig(file_name, format="pdf")

    def _optimize_posterior_mean(self) -> Tuple[Tensor, float]:
        """Optimizes the posterior mean function.

        Instead of implementing this functionality from scratch, simply use the
        exploitative acquisition function with BoTorch's optimization.

        Returns
        -------
        x_opt : torch.Tensor
            Location of the posterior mean function's optimum.
        f_opt : float
            Value of the posterior mean function's optimum.
        """
        if self.maximize:
            posterior_mean = PosteriorMean(model=self.gp)
        else:
            posterior_mean = NegativePosteriorMean(model=self.gp)

        x_opt, f_opt = optimize_acqf_botorch(
            posterior_mean,
            self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=2000,
            sequential=True,
        )
        x_opt = x_opt.squeeze(0)  # gets rid of superfluous dimension due to q=1
        f_opt = f_opt if self.maximize else -1 * f_opt

        # FIXME(lukasfro): Somehow something goes wrong with the standardization
        #  here... I could not make a minimum working example to reproduce this
        #  weird behaviour. Seems like outcome is de-normalized once too often.
        f_opt = self.gp.outcome_transform(f_opt)[0].squeeze().item()

        return x_opt, f_opt

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
        if not self.log_dir or self.gp is None:
            return

        # Saving GP model to file
        self.model_file = os.path.join(self.log_dir, "model_state.pth")
        torch.save(self.gp.state_dict(), self.model_file)

        # Save config to file
        self.config_file = os.path.join(self.log_dir, "config.yaml")
        yaml.dump(self.config, open(self.config_file, "w"))

        # Compute rolling best input/ouput pair
        x, y = self.data_handler.get_xy()

        # TODO(lukasfro): This should go into DataHandler class
        if self.maximize:
            idx = [torch.argmax(y[: i + 1]).item() for i in range(self.n_data)]
        else:
            idx = [torch.argmin(y[: i + 1]).item() for i in range(self.n_data)]
        x_best, y_best = x[idx], y[idx]

        # Update optimal parameters
        xn_opt, yn_opt = self.get_optimal_parameters()
        self.x_opt = torch.cat((self.x_opt, torch.atleast_2d(xn_opt)))
        self.y_opt = torch.cat((self.y_opt, torch.tensor([[yn_opt]])))

        # Store all and optimal evaluation inputs/outputs to file
        data = self.data_handler.get_xy(as_dict=True)
        data.update({"x_best": x_best, "y_best": y_best})
        data.update({"x_opt": self.x_opt, "y_opt": self.y_opt})
        data = {k: v.tolist() for k, v in data.items()}
        self.evaluations_file = os.path.join(self.log_dir, "evaluations.yaml")
        yaml.dump(data, open(self.evaluations_file, "w"), indent=2)
