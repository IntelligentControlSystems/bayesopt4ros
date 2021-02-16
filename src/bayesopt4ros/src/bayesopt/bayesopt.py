import numpy as np
import rospy
import yaml


class BayesianOptimization(object):
    """! The Bayesian optimization class.

    Implements the actual heavy lifting that is done behind the BayesOpt service.
    """
    def __init__(self, input_dim: int, max_iter: int, bounds: np.ndarray) -> None:
        """! The BayesianOptimization class initializer.

        @param input_dim    Number of input dimensions for the parameters.
        @param max_iter     Maximum number of iterations.
        @param bounds       Bounds specifying the optimization domain.
        """
        self.input_dim = input_dim
        self.max_iter = max_iter
        self.bounds = bounds
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
        )
