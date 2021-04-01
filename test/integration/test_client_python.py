#!/usr/bin/env python3

import actionlib
import itertools
import unittest
import numpy as np
import rospy
import rostest
import torch

from botorch.test_functions import SyntheticTestFunction, ThreeHumpCamel
from typing import Callable

from bayesopt4ros.msg import BayesOptAction, BayesOptGoal


class Forrester(SyntheticTestFunction):
    """The Forrester test function for global optimization.
    See definition here: https://www.sfu.ca/~ssurjano/forretal08.html
    """

    dim = 1
    _bounds = [(0.0, 1.0)]
    _optimal_value = -6.021
    _optimizers = [(0.757)]

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return (6.0 * X - 2.0) ** 2 * torch.sin(12.0 * X - 4.0)


class ShiftedThreeHumpCamel(ThreeHumpCamel):
    """The Three-Hump Camel test function for global optimization.

    See definition here: https://www.sfu.ca/~ssurjano/camel3.html

    .. note:: We shift the inputs as the initial design would hit the optimum directly.
    """

    _optimizers = [(0.5, 0.5)]

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return super().evaluate_true(X - 0.5)


class ExampleClient(object):
    """A demonstration on how to use the BayesOpt server from a Python node. """

    def __init__(self, server_name: str, objective: Callable, maximize=True) -> None:
        """Initializer of the client that queries the BayesOpt server.

        Parameters
        ----------
        server_name : str
            Name of the server (needs to be consistent with server node).
        objective : str
            Name of the example objective.
        maximize : bool
            If True, consider the problem a maximization problem.
        """
        rospy.init_node(self.__class__.__name__, anonymous=True, log_level=rospy.INFO)
        self.client = actionlib.SimpleActionClient(server_name, BayesOptAction)
        self.client.wait_for_server()
        self.server_name = server_name
        if objective == "Forrester":
            self.func = Forrester()
        elif objective == "NegativeForrester":
            self.func = Forrester(negate=True)
        elif objective == "ThreeHumpCamel":
            self.func = ShiftedThreeHumpCamel()
        else:
            raise ValueError("No such objective.")

        self.maximize = maximize
        self.y_best = -np.inf if maximize else np.inf
        self.x_best = None

    def request(self, y_new: float) -> np.ndarray:
        """Method that handles interaction with the server node.

        Parameters
        ----------
        value : float
            The function value obtained from the objective/experiment.

        Returns
        -------
        numpy.ndarray
            An array containing the new parameters suggested by BayesOpt server.
        """
        goal = BayesOptGoal(y_new=y_new)
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        return result.x_new

    def run(self) -> None:
        """Method that emulates client behavior."""
        # First value is just to trigger the server
        x_new = self.request(0.0)

        # Start querying the BayesOpt server until it reached max iterations
        for iter in itertools.count():
            rospy.loginfo(f"[Client] Iteration {iter + 1}")
            p_string = ", ".join([f"{xi:.3f}" for xi in x_new])
            rospy.loginfo(f"[Client] x_new = [{p_string}]")
            # Emulate experiment by querying the objective function
            y_new = self.func(torch.atleast_2d(torch.tensor(x_new))).squeeze().item()

            if (self.maximize and y_new > self.y_best) or (not self.maximize and y_new < self.y_best):
                self.y_best = y_new
                self.x_best = x_new

            rospy.loginfo(f"[Client] y_new = {y_new:.2f}, y_best = {self.y_best:.2f}")

            # Request server and obtain new parameters
            x_new = self.request(y_new)
            if not len(x_new):
                rospy.loginfo("[Client] Terminating - invalid response from server.")
                break


class ClientTestCase(unittest.TestCase):
    """Integration test cases for exemplary Python client."""

    _objective_name = None
    _maximize = True

    def test_objective(self) -> None:
        """Testing the client on the defined objective_function function. """
        node = ExampleClient(server_name="BayesOpt", objective=self._objective_name, maximize=self._maximize)
        node.run()

        x_opt = np.array(node.func.optimizers[0])
        y_opt = np.array(node.func.optimal_value)

        # Be kind w.r.t. precision of solution
        np.testing.assert_almost_equal(node.y_best, y_opt, decimal=2)
        np.testing.assert_almost_equal(node.x_best, x_opt, decimal=2)


class ClientTestCaseForrester(ClientTestCase):
    _objective_name = "Forrester"
    _maximize = False

class ClientTestCaseNegativeForrester(ClientTestCase):
    _objective_name = "NegativeForrester"
    _maximize = True

class ClientTestCaseThreeHumpCamel(ClientTestCase):
    _objective_name = "ThreeHumpCamel"
    _maximize = False


if __name__ == "__main__":
    # Note: unfortunately, rostest.rosrun does not allow to parse arguments
    # This can probably be done more efficiently but honestly, the ROS documentation for
    # integration testing is kind of outdated and not very thorough...
    objective = rospy.get_param("/objective")

    if objective == "Forrester":
        rostest.rosrun("bayesopt4ros", "test_python_client", ClientTestCaseForrester)
    elif objective == "NegativeForrester":
        rostest.rosrun("bayesopt4ros", "test_python_client", ClientTestCaseNegativeForrester)
    elif objective == "ThreeHumpCamel":
        rostest.rosrun(
            "bayesopt4ros", "test_python_client", ClientTestCaseThreeHumpCamel
        )
    else:
        raise ValueError("Not a known objective function.")
