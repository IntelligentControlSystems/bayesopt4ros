from __future__ import annotations

import os
import rospy
import torch
import yaml

from torch import Tensor
from typing import List, Tuple, Union, Optional

from botorch.acquisition import (
    AcquisitionFunction,
    UpperConfidenceBound,
    ExpectedImprovement,
)

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf as optimize_acqf_botorch
from botorch.optim.fit import fit_gpytorch_torch

from gpytorch.mlls import ExactMarginalLogLikelihood

from bayesopt4ros import util
from bayesopt4ros.data_handler import DataHandler
from bayesopt4ros.msg import BayesOptAction  # type: ignore
from bayesopt4ros.util import PosteriorMean

from gpytorch.means import Mean
from gpytorch.likelihoods.likelihood import Likelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform


USE_PRIOR_MEAN = True


class LearnedMean(Mean):
    def __init__(self, train_X, train_Y):
        super(LearnedMean, self).__init__()
        rospy.loginfo("PRIOR MEAN START FITTING")
        rospy.loginfo(train_X.shape)
        rospy.loginfo(train_Y.shape)
        self.gp = SingleTaskGP(train_X, train_Y)
        rospy.loginfo("PRIOR MEAN START FITTING 2")
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(self.mll)
        rospy.loginfo("PRIOR MEAN FITTED SUCCESSFULLY")
        # Important to turn off gradient computation for the inner GP
        self.gp.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            return self.gp.posterior(x).mean.detach().squeeze(-1)

    def predict_with_gp(self, x):
        with torch.no_grad():
            posterior = self.gp.posterior(x)
            mu = posterior.mean.detach()
            lower, upper = posterior.mvn.confidence_region()
            return lower.detach(), mu, upper.detach()


class PriorMeanGP(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        mean_module: Mean,
        likelihood: Optional[Likelihood] = None,
        covar_module=None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        super().__init__(
            train_X,
            train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.mean_module = mean_module


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
        debug_visualization: bool = False,
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
        debug_visualization : bool
            If True, the optimization of the acquisition function is visualized.
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
        self.debug_visualization = debug_visualization
        self.data_handler = DataHandler(maximize=self.maximize)
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

    def update_last_goal(self, goal: BayesOptAction) -> None:
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
        """Get the optimal parameters with corresponding expected value.

        .. note:: 'Optimal' referes to the optimum of the GP model.

        Returns
        -------
        torch.Tensor
            Location of the GP posterior mean's optimum.
        float
            Function value of the GP posterior mean's optium.

        See Also
        --------
        get_best_observation
        """
        return self._optimize_posterior_mean()

    def get_best_observation(self) -> Tuple[torch.Tensor, float]:
        """Get the best parameters and corresponding observed value.

        .. note:: 'Best' refers to the highest/lowest observed datum.

        Returns
        -------
        torch.Tensor
            Location of the highest/lowest observed datum.
        float
            Function value of the highest/lowest observed datum.

        See Also
        --------
        get_optimal_parameters
        """
        return self.data_handler.x_best, self.data_handler.y_best

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

    def _get_next_x(self) -> Tensor:
        """Computes the next point to evaluate.

        Returns
        -------
        torch.Tensor
            The next point to evaluate.
        """
        if self.n_data < self.n_init:  # We are in the initialization phase
            x_new = self.x_init[self.n_data]
        else:  # Actually optimizing the acquisition function for new points
            x_new = self._optimize_acqf(
                self._initialize_acqf(), visualize=self.debug_visualization
            )[0]

            # To avoid numerical issues and encourage exploration
            # if self._check_data_vicinity(x_new, self.data_handler.get_xy()[0]):
            #     rospy.logwarn("[BayesOpt] x_new is too close to existing data.")
            #     lb, ub = self.bounds[0], self.bounds[1]
            #     x_rand = lb + (ub - lb) * torch.rand((self.input_dim,))
            #     x_new = x_rand

        return x_new

    def _check_config(self, load_dirs: List[str]):
        """Make sure that all relevant parameters in the configs match.

        Parameters
        ----------
        load_dirs : str or List[str]
            The directories to the previous experiments, which are loaded.
        """
        load_dirs = [load_dirs] if isinstance(load_dirs, str) else load_dirs
        for load_dir in load_dirs:
            with open(os.path.join(load_dir, "config.yaml")) as f:
                load_config = yaml.load(f, Loader=yaml.FullLoader)

            for p in self.constant_config_parameters:
                try:
                    assert load_config[p] == self.__getattribute__(p)
                except AssertionError:
                    rospy.logerr(f"Your configuration does not match with {load_dir}")

    def _load_prev_bayesopt(
        self, load_dirs: Union[str, List[str]]
    ) -> Tuple[DataHandler, GPyTorchModel]:
        """Load data from previous BO experiments.

        Parameters
        ----------
        load_dirs : str or List[str]
            The directories to the previous experiments, which are loaded.

        Returns
        -------
        :class:`DataHandler`
            An data handler object with filled with observations from previous
            experiments.
        :class:`GPyTorchModel`
            A GP object that has been trained on the data from previous experiments.
        """
        # We can load multiple previous runs
        load_dirs = [load_dirs] if isinstance(load_dirs, str) else load_dirs

        # Configurations need to be compatible with the current one
        self._check_config(load_dirs)

        # Create model with the previous runs' data
        data_files = [
            os.path.join(load_dir, "evaluations.yaml") for load_dir in load_dirs
        ]
        self.data_handler = DataHandler.from_file(data_files)
        self.data_handler.maximize = self.maximize

        if USE_PRIOR_MEAN:
            # We use previous data to fit an informed prior mean function,
            # Here, we just fit the inner GP. The surrogate model GP is only
            # created after we have observed new data, just as in the standard
            # setting. The data handler is also re-set.
            
            x, y = self.data_handler.get_xy()
            rospy.loginfo("START PRIOR MEAN")

            self.mean_module = LearnedMean(x, y)

            # As if we haven't observed data yet
            self.data_handler = DataHandler(maximize=self.maximize)
            self.gp = None
        else:
            self.gp = self._initialize_model(self.data_handler)
            self._fit_model()

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

        # Note: We always create a GP model from scratch when receiving new data.
        # The reason is the following: if the 'set_train_data' method of the GP
        # is used instead, the normalization/standardization of the input/output
        # data is not updated in the GPyTorchModel. We also want at least 2 data
        # points such that the input normalization works properly.
        self.data_handler.add_xy(x=self.x_new, y=goal.y_new)
        self.gp = self._initialize_model(self.data_handler)
        self._fit_model()

    def _initialize_model(self, data_handler: DataHandler) -> GPyTorchModel:
        """Creates a GP object from data.

        .. note:: Currently the kernel types are hard-coded. However, Matern is
            a good default choice.

        Parameters
        ----------
        :class:`DataHandler`
            A data handler object containing the observations to create the model.

        Returns
        -------
        :class:`GPyTorchModel`
            A GP object.
        """
        x, y = data_handler.get_xy()

        if x.shape[0] < 2:
            rospy.logerr("[BO] DO NOT USE TRANSFORMATION")
            outcome_transform = None
            input_transform = None
        else:
            rospy.logerr("[BO] DO USE TRANSFORMATION")
            outcome_transform = Standardize(m=1)
            input_transform = Normalize(d=self.input_dim, bounds=self.bounds)

        if USE_PRIOR_MEAN:
            gp = PriorMeanGP(
                train_X=x,
                train_Y=y,
                mean_module=self.mean_module,
                outcome_transform=outcome_transform,
                input_transform=input_transform,
            )
        else:
            gp = SingleTaskGP(
                train_X=x,
                train_Y=y,
                outcome_transform=outcome_transform,
                input_transform=input_transform,
            )
        return gp

    def _fit_model(self) -> None:
        """Performs inference and fits the GP to the data."""
        # Scipy optimizers is faster and more accurate but tends to be numerically
        # less table for single precision. To avoid error checking, we use the
        # stochastic optimizer.
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options={"disp": False})

    def _initialize_acqf(self) -> AcquisitionFunction:
        """Initialize the acquisition function of choice.

        Returns
        -------
        :class:`AcquisitionFunction`
            An acquisition function based on BoTorch's base class.
        """
        if self.acq_func.upper() == "UCB":
            acq_func = UpperConfidenceBound(
                model=self.gp, beta=4.0, maximize=self.maximize
            )
        elif self.acq_func.upper() == "EI":
            best_f = self.data_handler.y_best  # note that EI assumes noiseless
            acq_func = ExpectedImprovement(
                model=self.gp, best_f=best_f, maximize=self.maximize
            )
        elif self.acq_func.upper() == "NEI":
            raise NotImplementedError(
                "Not implemented yet. Always leads to numerical issues"
            )
        else:
            raise NotImplementedError(
                f"{self.acq_func} is not a valid acquisition function"
            )
        return acq_func

    def _optimize_acqf(
        self, acq_func: AcquisitionFunction, visualize: bool = False
    ) -> Tuple[Tensor, float]:
        """Optimizes the acquisition function.

        Parameters
        ----------
        acq_func : :class:`AcquisitionFunction`
            The acquisition function to optimize.
        visualize : bool
            Flag if debug visualization should be turned on.

        Returns
        -------
        x_opt : torch.Tensor
            Location of the acquisition function's optimum.
        f_opt : float
            Value of the acquisition function's optimum.
        """
        x_opt, f_opt = optimize_acqf_botorch(
            acq_func,
            self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=2000,
            sequential=True,
        )

        if visualize:
            self._debug_acqf_visualize(acq_func, x_opt, f_opt)

        x_opt = x_opt.squeeze(0)  # gets rid of superfluous dimension due to q=1
        return x_opt, f_opt

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
        posterior_mean = PosteriorMean(model=self.gp, maximize=self.maximize)
        x_opt, f_opt = self._optimize_acqf(posterior_mean)
        f_opt = f_opt if self.maximize else -1 * f_opt
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

    def _check_data_vicinity(self, x1, x2):
        """Returns true if `x1` is close to any point in `x2`.

        .. note:: We are following Binois and Picheny (2019) and check if the
            proposed point is too close to any existing data points to avoid
            numerical issues. In that case, choose a random point instead.
            https://www.jstatsoft.org/article/view/v089i08

        Parameters
        ----------
        x1 : torch.Tensor
            A single data point.
        x2 : torch.Tensor
            Multiple data points.

        Returns
        -------
        bool
            Returns `True` if `x1` is close to any point in `x2` else returns `False`
        """
        x1 = torch.atleast_2d(x1)
        assert x1.shape[0] == 1
        X = torch.cat((x2, x1))
        c = self.gp.posterior(X).mvn.covariance_matrix
        cis = c[-1, :-1]
        cii = c.diag()[:-1]
        css = c.diag()[-1]
        kss = self.gp.covar_module.outputscale
        d = torch.min((cii + css - 2 * cis) / kss)
        return d < 1e-5

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
        x_best = self.data_handler.x_best_accumulate
        y_best = self.data_handler.y_best_accumulate

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

    def _debug_acqf_visualize(self, acq_func, x_opt, f_opt):
        """Visualize the acquisition function for debugging purposes."""
        import matplotlib.pyplot as plt

        if self.input_dim not in [1, 2]:
            return
        elif self.input_dim == 1:
            # The plotting ranges
            lb, ub = self.bounds[0], self.bounds[1]
            xs = torch.linspace(lb.item(), ub.item(), 500).unsqueeze(-1)

            # Evaluate GP and acquisition function
            posterior = self.gp.posterior(xs, observation_noise=False)
            mean = posterior.mean.squeeze().detach()
            std = posterior.variance.sqrt().squeeze().detach()
            acqf = acq_func(xs.unsqueeze(1).unsqueeze(1)).squeeze().detach()
            x_eval, y_eval = self.data_handler.get_xy()

            # Create plot
            _, axes = plt.subplots(nrows=2, ncols=1)
            axes[0].plot(xs, mean, label="GP mean")
            axes[0].fill_between(
                xs.squeeze(), mean + 2 * std, mean - 2 * std, alpha=0.3
            )
            axes[0].plot(x_eval, y_eval, "ko")
            axes[0].grid()

            axes[1].plot(xs, acqf)
            axes[1].plot(x_opt, f_opt, "C3x")
            axes[1].grid()
        elif self.input_dim == 2:
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

            # Create plot
            _, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].contourf(x1, x2, gpm, levels=50)
            axes[0].plot(x_eval[:, 0], x_eval[:, 1], "ko")
            axes[0].plot(x_opt[0, 0], x_opt[0, 1], "C3x")
            axes[0].axis("equal")
            c = axes[1].contourf(x1, x2, acqf, levels=50)
            axes[1].plot(x_opt[0, 0], x_opt[0, 1], "C3x")
            axes[1].axis("equal")

            plt.colorbar(c)

        plt.tight_layout()
        file_name = os.path.join(self.log_dir, f"acqf_visualize_{x_eval.shape[0]}.pdf")
        rospy.logdebug(f"Saving debug visualization to: {file_name}")
        plt.savefig(file_name, format="pdf")
        plt.close()
