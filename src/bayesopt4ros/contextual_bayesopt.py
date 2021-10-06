from __future__ import annotations

import torch
import yaml

from torch import Tensor
from typing import Tuple, List

from botorch.models import SingleTaskGP
from botorch.acquisition import AcquisitionFunction, FixedFeatureAcquisitionFunction
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

from bayesopt4ros import BayesianOptimization
from bayesopt4ros.data_handler import DataHandler
from bayesopt4ros.util import PosteriorMean


class ContextualBayesianOptimization(BayesianOptimization):
    """The contextual Bayesian optimization class.

    Implements the actual heavy lifting that is done under the hood of
    :class:`contextual_bayesopt_server.ContextualBayesOptServer`.

    See Also
    --------
    :class:`bayesopt.BayesianOptimization`
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
        # Needs to be initialized before super() is called to check config in load_dir
        self.context_dim = context_dim
        self.context, self.prev_context = None, None
        self.joint_dim = input_dim + self.context_dim

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

    @classmethod
    def from_file(cls, config_file: str) -> ContextualBayesianOptimization:
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

    def get_optimal_parameters(self, context=None) -> Tuple[torch.Tensor, float]:
        """Get the optimal parameters for given context with corresponding value.

        .. note:: 'Optimal' referes to the optimum of the GP model.

        Parameters
        ----------
        context : torch.Tensor, optional
            The context for which to get the optimal parameters. If none is
            none is provided, use the last observed context.

        Returns
        -------
        torch.Tensor
            Location of the GP posterior mean's optimum for the given context.
        float
            Function value of the GP posterior mean's optium for the given context.

        See Also
        --------
        get_best_observation
        """
        return self._optimize_posterior_mean(context)

    def get_best_observation(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Get the best parameters, context and corresponding observed value.

        .. note:: 'Best' refers to the highest/lowest observed datum.

        Returns
        -------
        torch.Tensor
            Location of the highest/lowest observed datum.
        torch.Tensor
            Context of the highest/lowest observed datum.
        float
            Function value of the highest/lowest observed datum.

        See Also
        --------
        get_optimal_parameters
        """
        x_best, c_best = torch.split(
            self.data_handler.x_best, [self.input_dim, self.context_dim]
        )
        return x_best, c_best, self.data_handler.y_best

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
            self.prev_context = self.context
            return

        # Concatenate context and optimization variable
        x = torch.cat((self.x_new, self.context))
        self.data_handler.add_xy(x=x, y=goal.y_new)
        self.prev_context = self.context
        self.context = torch.tensor(goal.c_new)

        # Note: We always create a GP model from scratch when receiving new data.
        # The reason is the following: if the 'set_train_data' method of the GP
        # is used instead, the normalization/standardization of the input/output
        # data is not updated in the GPyTorchModel. We also want at least 2 data
        # points such that the input normalization works properly.
        if self.data_handler.n_data >= 2:
            self.gp = self._initialize_model(self.data_handler)
            self._fit_model()

    def _initialize_model(self, data_handler: DataHandler) -> GPyTorchModel:
        """Creates a GP object from data.

        .. note:: Currently the kernel types are hard-coded. However, Matern is
            a good default choice. The joint kernel is just the multiplication
            of the parameter and context kernels.

        Parameters
        ----------
        :class:`DataHandler`
            A data handler object containing the observations to create the model.

        Returns
        -------
        :class:`GPyTorchModel`
            A GP object.
        """
        # Kernel for optimization variables
        ad0 = tuple(range(self.input_dim))
        k0 = MaternKernel(active_dims=ad0, lengthscale_prior=GammaPrior(3.0, 6.0))

        # Kernel for context variables
        ad1 = tuple(range(self.input_dim, self.input_dim + self.context_dim))
        k1 = MaternKernel(active_dims=ad1, lengthscale_prior=GammaPrior(3.0, 6.0))

        # Joint kernel is constructed via multiplication
        covar_module = ScaleKernel(k0 * k1, outputscale_prior=GammaPrior(2.0, 0.15))

        # For contextual BO, we do not want to specify the bounds for the context
        # variables (who knows what they might be...). We therefore use the neat
        # feature of BoTorch to infer the normalization bounds from data. However,
        # this does not work is only a single data point is given.
        input_transform = (
            Normalize(d=self.joint_dim) if len(self.data_handler) > 1 else None
        )
        x, y = data_handler.get_xy()
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            outcome_transform=Standardize(m=1),
            input_transform=input_transform,
            covar_module=covar_module,
        )
        return gp

    def _initialize_acqf(self) -> FixedFeatureAcquisitionFunction:
        """Initialize the acquisition function of choice and wrap it with the
        FixedFeatureAcquisitionFunction given the current context.

        Returns
        -------
        FixedFeatureAcquisitionFunction
            An acquisition function of choice with fixed features.
        """
        acq_func = super()._initialize_acqf()
        columns = [i + self.input_dim for i in range(self.context_dim)]
        values = self.context.tolist()
        acq_func_ff = FixedFeatureAcquisitionFunction(
            acq_func, d=self.joint_dim, columns=columns, values=values
        )
        return acq_func_ff

    def _optimize_acqf(
        self, acq_func: AcquisitionFunction, visualize: bool = False
    ) -> Tuple[Tensor, float]:
        """Optimizes the acquisition function with the context variable fixed.

        Note: The debug visualization is turned off for contextual setting.

        Parameters
        ----------
        acq_func : AcquisitionFunction
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
        x_opt, f_opt = super()._optimize_acqf(acq_func, visualize=False)
        if visualize:
            pass
        return x_opt, f_opt

    def _optimize_posterior_mean(self, context=None) -> Tuple[Tensor, float]:
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
        x_opt : torch.Tensor
            Location of the posterior mean function's optimum.
        f_opt : float
            Value of the posterior mean function's optimum.
        """
        context = context or self.prev_context
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context)

        columns = [i + self.input_dim for i in range(self.context_dim)]
        values = context.tolist()

        pm = PosteriorMean(model=self.gp, maximize=self.maximize)
        pm_ff = FixedFeatureAcquisitionFunction(pm, self.joint_dim, columns, values)

        x_opt, f_opt = super()._optimize_acqf(pm_ff, visualize=False)
        f_opt = f_opt if self.maximize else -1 * f_opt
        return x_opt, f_opt

    def _check_data_vicinity(self, x1, x2):
        """Returns true if `x1` is close to any point in `x2`.

        .. note:: We are following Binois and Picheny (2019) and check if the
            proposed point is too close to any existing data points to avoid
            numerical issues. In that case, choose a random point instead.
            https://www.jstatsoft.org/article/view/v089i08

        .. note:: `x1` is considered without context whereas `x2` contains the
            context. The reasons for that is to have a consistent interface
            with the standard BO implementation.

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
        xc1 = torch.cat((x1, self.context))
        return super()._check_data_vicinity(xc1, x2)
