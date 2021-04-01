import rospy
import torch
import yaml

from torch import Tensor

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
        config: dict = None,
        maximize : bool = True,
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
            config=config,
            maximize=maximize,
        )
        self.context_dim = context_dim

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
            log_dir=config["log_dir"],
            maximize=config["maximize"],
            config=config,
        )

    def _update_model(self, goal):
        raise NotImplementedError()

    def _initialize_model(self, goal):
        raise NotImplementedError()

    def _optimize_acq(self):
        raise NotImplementedError()    
