from __future__ import annotations

import rospy
import torch
import yaml

from torch import Tensor
from typing import Tuple, List

from botorch.acquisition import PosteriorMean
from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf as optimize_acqf_botorch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

from bayesopt4ros import BayesianOptimization
from bayesopt4ros.util import NegativePosteriorMean


class ContextualBayesianOptimization(BayesianOptimization):
    """The contextual Bayesian optimization class.

    Implements the actual heavy lifting that is done under the hood of
    :class:`contextual_bayesopt_server.ContextualBayesOptServer`.

    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        max_iter: int,
        bounds: Tensor,
        context_bounds: Tensor,
        acq_func: str = "UCB",
        n_init: int = 5,
        log_dir: str = None,
        load_dir: str = None,
        config: dict = None,
        maximize: bool = True,
    ) -> None:
        """The ContextualBayesianOptimization class initializer.

        .. note:: For a definition of the other arguments, see
            :class:`bayesopt.BayesianOptimization`.

        Parameters
        ----------
        context_dim : int
            Number of context dimensions for the parameters.
        context_bounds : torch.Tensor
            A [2, context_dim] shaped tensor specifying the context variables domain.
        """
        super().__init__(
            input_dim=input_dim,
            max_iter=max_iter,
            bounds=bounds,
            acq_func=acq_func,
            n_init=n_init,
            log_dir=log_dir,
            load_dir=load_dir,
            config=config,
            maximize=maximize,
        )
        self.context = None
        self.context_dim = context_dim
        self.context_bounds = context_bounds
        self.joint_dim = self.input_dim + self.context_dim
        self.joint_bounds = torch.cat((self.bounds, self.context_bounds), dim=1)

    @classmethod
    def from_file(cls, config_file: str) -> ContextualBayesianOptimization:
        # TODO(lukasfro): Does not feel right to copy that much code from base class
        """Initialize a ContextualBayesianOptimization instance from a config file.

        Parameters
        ----------
        config_file : str
            The config file (full path, relative or absolute).

        Returns
        -------
        :class:`ContextualBayesianOptimization`
            An instance of the ContextualBayesianOptimization class.
        """
        # Read config from file
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Bring bounds in correct format
        lb = torch.tensor(config["lower_bound"])
        ub = torch.tensor(config["upper_bound"])
        bounds = torch.stack((lb, ub))

        lbc = torch.tensor(config["lower_bound_context"])
        ubc = torch.tensor(config["upper_bound_context"])
        context_bounds = torch.stack((lbc, ubc))

        # Construct class instance based on the config
        return cls(
            input_dim=config["input_dim"],
            context_dim=config["context_dim"],
            max_iter=config["max_iter"],
            bounds=bounds,
            context_bounds=context_bounds,
            acq_func=config["acq_func"],
            n_init=config["n_init"],
            log_dir=config.get("log_dir"),
            load_dir=config.get("load_dir"),
            maximize=config["maximize"],
            config=config,
        )

    def get_best_observation(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Get the best parameters, context and corresponding observed value."""
        x_best, c_best = torch.split(self.x_best, [self.input_dim, self.context_dim])
        return x_best, c_best, self.y_best

    def get_optimal_parameters(self, context) -> Tuple[torch.Tensor, float]:
        """Geth the optimal parameters for given context with corresponding value."""
        return self._optimize_posterior_mean(context)

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
        return ["input_dim", "context_dim", "maximize"]

    def _update_model(self, goal):
        """Updates the GP with new data as well as the current context. Creates
        a model if none exists yet.

        Parameters
        ----------
        goal : ContextualBayesOptAction
            The goal (context variable of the current goal is always pre-ceding
            the function value, i.e., the goal consists of [y_n, c_{n+1}]) sent
            from the client for the most recent experiment.
        """
        if self.x_new is None and self.context is None:
            # The very first function value we obtain from the client is just to
            # trigger the server. At that point, there is no new input point,
            # hence, no need to need to update the model. However, the initial
            # context is already valid.
            self.context = torch.tensor(goal.c_new)
            return

        # Concatenate context and optimization variable
        x = torch.cat((self.x_new, self.context))
        self.data_handler.add_xy(x=x, y=goal.y_new)
        self.context = torch.tensor(goal.c_new)

        if self.n_data >= self.n_init:
            # Only create model once we are done with the initial design phase
            if self.gp:
                x, y = self.data_handler.get_xy()
                self.gp.set_train_data(x, y.squeeze(-1), strict=False)
            else:
                self.gp = self._initialize_model(*self.data_handler.get_xy())
            self._optimize_model()

    def _initialize_model(self, x, y) -> GPyTorchModel:
        # Kernel for optimization variables
        ad0 = tuple(range(self.input_dim))
        k0 = MaternKernel(active_dims=ad0, lengthscale_prior=GammaPrior(3.0, 6.0))

        # Kernel for context variables
        ad1 = tuple(range(self.input_dim, self.input_dim + self.context_dim))
        k1 = MaternKernel(active_dims=ad1, lengthscale_prior=GammaPrior(3.0, 6.0))

        # Joint kernel is constructed via multiplication
        covar_module = ScaleKernel(k0 * k1, outputscale_prior=GammaPrior(2.0, 0.15))

        unit_cube = torch.stack((torch.zeros(self.joint_dim), torch.ones(self.joint_dim)))
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(d=self.joint_dim, bounds=unit_cube),
            covar_module=covar_module,
        )
        return gp

    def _optimize_acqf(self) -> Tensor:
        """Optimizes the acquisition function with the context variable fixed.

        Returns
        -------
        torch.Tensor
            Location of the acquisition function's optimum (without context).
        """
        # TODO(lukasfro): Re-factor using botorch.FixedFeatureAcquisitionFunction
        acq_func = self._initialize_acqf()
        fixed_features = {i + self.input_dim: self.context[i] for i in range(self.context_dim)}
        x_opt, _ = optimize_acqf_botorch(
            acq_func,
            self.joint_bounds,
            q=1,
            num_restarts=10,
            raw_samples=2000,
            sequential=True,
            fixed_features=fixed_features,
        )
        x_opt = x_opt.squeeze(0)  # gets rid of superfluous dimension due to q=1
        x_opt = x_opt[: self.input_dim]  # only return the next input parameters
        return x_opt

    def _optimize_posterior_mean(self, context=None) -> Tensor:
        """Optimizes the posterior mean function with a fixed context variable.

        Instead of implementing this functionality from scratch, simply use the
        exploitative acquisition function with BoTorch's optimization.

        Parameters
        ----------
        context : torch.Tensor, optional
            The context for which to compute the mean's optimum. If none is
            specified, use the last one that was received.

        Returns
        -------
        torch.Tensor
            Location of the posterior mean function's optimum (without context).
        """
        # TODO(lukasfro): Re-factor once the PR is through
        if self.maximize:
            posterior_mean = PosteriorMean(model=self.gp)
        else:
            posterior_mean = NegativePosteriorMean(model=self.gp)

        # TODO(lukasfro): Re-factor acqf optimization. We have this piece of code 3x by now...
        context = context or self.context
        fixed_features = {i + self.input_dim: context[i] for i in range(self.context_dim)}
        x_opt, f_opt = optimize_acqf_botorch(
            posterior_mean,
            self.joint_bounds,
            q=1,
            num_restarts=10,
            raw_samples=2000,
            sequential=True,
            fixed_features=fixed_features,
        )
        x_opt = x_opt.squeeze(0)  # gets rid of superfluous dimension due to q=1
        x_opt = x_opt[: self.input_dim]  # only return the next input parameters
        f_opt = f_opt if self.maximize else -1 * f_opt

        # FIXME(lukasfro): Somehow something goes wrong with the standardization
        #  here... I could not make a minimum working example to reproduce this
        #  weird behaviour. Seems like outcome is de-normalized once too often.
        f_opt = self.gp.outcome_transform(f_opt)[0].squeeze().item()

        return x_opt, f_opt
