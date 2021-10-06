#!/usr/bin/env python3

import actionlib
import itertools
import unittest
import numpy as np
import rospy
import rostest
import torch

from typing import Callable

from bayesopt4ros import test_objectives
from bayesopt4ros.msg import BayesOptAction, BayesOptGoal
from bayesopt4ros.msg import BayesOptStateAction, BayesOptStateGoal, BayesOptStateResult


class ExampleClient(object):
    """A demonstration on how to use the BayesOpt server from a Python node."""

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
        if objective == "Forrester":
            self.func = test_objectives.Forrester()
        elif objective == "NegativeForrester":
            self.func = test_objectives.Forrester(negate=True)
        elif objective == "NoisyForrester":
            self.func = test_objectives.Forrester(noise_std=0.5)
        elif objective == "ThreeHumpCamel":
            self.func = test_objectives.ShiftedThreeHumpCamel()
        else:
            raise ValueError("No such objective.")
        self.maximize = maximize

    def request_parameter(self, y_new: float) -> np.ndarray:
        """Method that requests new parameters from the BayesOpt server.

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

    def request_bayesopt_state(self) -> BayesOptStateResult:
        """Method that requests the (final) state of BayesOpt server.

        .. note:: As we only call this function once, we can just create the
            corresponding client locally.
        """
        state_client = actionlib.SimpleActionClient(
            "BayesOptState", BayesOptStateAction
        )
        state_client.wait_for_server()

        goal = BayesOptStateGoal()
        state_client.send_goal(goal)
        state_client.wait_for_result()
        return state_client.get_result()

    def run(self) -> None:
        """Method that emulates client behavior."""
        # First value is just to trigger the server
        x_new = self.request_parameter(0.0)

        # Start querying the BayesOpt server until it reached max iterations
        for iter in itertools.count():
            rospy.loginfo(f"[Client] Iteration {iter + 1}")
            p_string = ", ".join([f"{xi:.3f}" for xi in x_new])
            rospy.loginfo(f"[Client] x_new = [{p_string}]")

            # Emulate experiment by querying the objective function
            y_new = self.func(torch.atleast_2d(torch.tensor(x_new))).squeeze().item()
            rospy.loginfo(f"[Client] y_new = {y_new:.2f}")

            # Request server and obtain new parameters
            x_new = self.request_parameter(y_new)
            if not len(x_new):
                rospy.loginfo("[Client] Terminating - invalid response from server.")
                break


class ClientTestCase(unittest.TestCase):
    """Integration test cases for exemplary Python client."""

    _objective_name = None
    _maximize = True

    def test_objective(self) -> None:
        """Testing the client on the defined objective function."""

        # Set up the client
        node = ExampleClient(
            server_name="BayesOpt",
            objective=self._objective_name,
            maximize=self._maximize,
        )

        # Emulate experiment
        node.run()

        # Get the (estimated) optimum of the objective
        result = node.request_bayesopt_state()

        # True optimum of the objective
        x_opt = np.array(node.func.optimizers[0])
        f_opt = np.array(node.func.optimal_value)

        # Be kind w.r.t. precision of solution
        np.testing.assert_almost_equal(result.x_opt, x_opt, decimal=1)
        np.testing.assert_almost_equal(result.f_opt, f_opt, decimal=1)


class ClientTestCaseForrester(ClientTestCase):
    _objective_name = "Forrester"
    _maximize = False


class ClientTestCaseNegativeForrester(ClientTestCase):
    _objective_name = "NegativeForrester"
    _maximize = True


class ClientTestCaseNoisyForrester(ClientTestCase):
    _objective_name = "NoisyForrester"
    _maximize = False


class ClientTestCaseThreeHumpCamel(ClientTestCase):
    _objective_name = "ThreeHumpCamel"
    _maximize = False


if __name__ == "__main__":
    # Note: unfortunately, rostest.rosrun does not allow to parse arguments
    # This can probably be done more efficiently but honestly, the ROS documentation for
    # integration testing is kind of outdated and not very thorough...

    objective = rospy.get_param("/objective")
    rospy.logwarn(f"Objective: {objective}")
    if objective == "Forrester":
        rostest.rosrun("bayesopt4ros", "test_python_client", ClientTestCaseForrester)
    elif objective == "NegativeForrester":
        rostest.rosrun(
            "bayesopt4ros", "test_python_client", ClientTestCaseNegativeForrester
        )
    elif objective == "NoisyForrester":
        rostest.rosrun(
            "bayesopt4ros", "test_python_client", ClientTestCaseNoisyForrester
        )
    elif objective == "ThreeHumpCamel":
        rostest.rosrun(
            "bayesopt4ros", "test_python_client", ClientTestCaseThreeHumpCamel
        )
    else:
        raise ValueError("Not a known objective function.")
