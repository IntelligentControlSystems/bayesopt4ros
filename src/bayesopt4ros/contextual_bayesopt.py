from __future__ import annotations

import rospy
import torch
import yaml

from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf as optimize_acqf_botorch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

from bayesopt4ros import BayesianOptimization


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
        self.context_dim = context_dim
        self.context = None

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

        # Construct class instance based on the config
        return cls(
            input_dim=config["input_dim"],
            context_dim=config["context_dim"],
            max_iter=config["max_iter"],
            bounds=bounds,
            acq_func=config["acq_func"],
            n_init=config["n_init"],
            log_dir=config.get("log_dir"),
            load_dir=config.get("load_dir"),
            maximize=config["maximize"],
            config=config,
        )

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
            self.context = goal.c_new
            return

        # Concatenate context and optimization variable
        x = torch.cat((self.x_new, self.context))
        self.data_handler.add_xy(x=x, y=torch.tensor([[goal.y_new]]))
        self.context = goal.c_new

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
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(d=self.input_dim),
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
        acq_func = self._initialize_acqf()
        context_idx = range(self.input_dim, self.input_dim + self.context_dim)
        fixed_features = {i + self.input_dim: self.context[i] for i in context_idx}
        x_opt, _ = optimize_acqf_botorch(
            acq_func,
            self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=2000,
            sequential=True,
            fixed_features=fixed_features,
        )
        x_opt = x_opt.squeeze(0)  # gets rid of superfluous dimension due to q=1
        x_opt = x_opt[:self.input_dim]  # only return the next input parameters
        return x_opt
