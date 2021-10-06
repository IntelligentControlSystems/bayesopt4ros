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
from bayesopt4ros.msg import ContextualBayesOptAction, ContextualBayesOptGoal
from bayesopt4ros.msg import (
    ContextualBayesOptStateAction,
    ContextualBayesOptStateGoal,
    ContextualBayesOptStateResult,
)


class ExampleContextualClient(object):
    """A demonstration on how to use the contexutal BayesOpt server from a Python
    node."""

    def __init__(self, server_name: str, objective: Callable, maximize=True) -> None:
        """Initializer of the client that queries the contextual BayesOpt server.

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
        self.client = actionlib.SimpleActionClient(
            server_name, ContextualBayesOptAction
        )

        self.client.wait_for_server()

        if objective == "ContextualForrester":
            self.func = test_objectives.ContextualForrester()
        else:
            raise ValueError("No such objective.")

        self.maximize = maximize
        self.y_best = -np.inf if maximize else np.inf
        self.x_best = None

    def request_parameter(self, y_new: float, c_new: np.ndarray) -> np.ndarray:
        """Method that requests new parameters from the ContextualBayesOpt
        server for a given context.

        Parameters
        ----------
        y_new : float
            The function value obtained from the objective/experiment.
        c_new : np.ndarray
            The context variable for the next evaluation/experiment.

        Returns
        -------
        numpy.ndarray
            An array containing the new parameters suggested by contextual BayesOpt
            server.
        """
        goal = ContextualBayesOptGoal(y_new=y_new, c_new=c_new)
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        return torch.tensor(result.x_new)

    def request_bayesopt_state(self, context) -> ContextualBayesOptStateResult:
        """Method that requests the (final) state of BayesOpt server.

        .. note:: As we only call this function once, we can just create the
            corresponding client locally.
        """
        state_client = actionlib.SimpleActionClient(
            "ContextualBayesOptState", ContextualBayesOptStateAction
        )
        state_client.wait_for_server()

        goal = ContextualBayesOptStateGoal()
        goal.context = list(context)
        state_client.send_goal(goal)
        state_client.wait_for_result()
        return state_client.get_result()

    def run(self) -> None:
        """Method that emulates client behavior."""
        # First value is just to trigger the server

        c_new = self.sample_context()
        x_new = self.request_parameter(y_new=0.0, c_new=c_new)

        # Start querying the BayesOpt server until it reached max iterations
        for iter in itertools.count():
            rospy.loginfo(f"[Client] Iteration {iter + 1}")
            x_string = ", ".join([f"{xi:.3f}" for xi in x_new])
            c_string = ", ".join([f"{xi:.3f}" for xi in c_new])
            rospy.loginfo(f"[Client] x_new = [{x_string}] for c_new = [{c_string}]")

            # Emulate experiment by querying the objective function
            xc_new = torch.atleast_2d(torch.cat((x_new, c_new)))
            y_new = self.func(xc_new).squeeze().item()

            if (self.maximize and y_new > self.y_best) or (
                not self.maximize and y_new < self.y_best
            ):
                self.y_best = y_new
                self.x_best = x_new

            rospy.loginfo(f"[Client] y_new = {y_new:.2f}")

            # Request server and obtain new parameters
            c_new = self.sample_context()
            x_new = self.request_parameter(y_new=y_new, c_new=c_new)
            if not len(x_new):
                rospy.loginfo("[Client] Terminating - invalid response from server.")
                break

    def sample_context(self) -> np.ndarray:
        """Samples a random context variable to emulate the client."""
        context_bounds = [b for b in self.func._bounds[self.func.input_dim :]]
        context = torch.tensor([np.random.uniform(b[0], b[1]) for b in context_bounds])
        return context


class ContextualClientTestCase(unittest.TestCase):
    """Integration test cases for exemplary contextual Python client."""

    _objective_name = None
    _maximize = True

    def test_objective(self) -> None:
        """Testing the client on the defined objective function and couple of
        contexts."""

        # Set up the client
        node = ExampleContextualClient(
            server_name="ContextualBayesOpt",
            objective=self._objective_name,
            maximize=self._maximize,
        )

        # Emulate experiment
        node.run()

        # Check the estimated optimum for different contexts
        for context, x_opt, f_opt in zip(
            node.func._test_contexts,
            node.func._contextual_optimizer,
            node.func._contextual_optimal_values,
        ):
            # Get the (estimated) optimum of the objective for a given context
            result = node.request_bayesopt_state(context)

            # Be kind w.r.t. precision of solution
            np.testing.assert_almost_equal(result.x_opt, x_opt, decimal=1)
            np.testing.assert_almost_equal(result.f_opt, f_opt, decimal=1)


class ContextualClientTestCaseForrester(ContextualClientTestCase):
    _objective_name = "ContextualForrester"
    _maximize = False


if __name__ == "__main__":
    # Note: unfortunately, rostest.rosrun does not allow to parse arguments
    # This can probably be done more efficiently but honestly, the ROS documentation for
    # integration testing is kind of outdated and not very thorough...
    objective = rospy.get_param("/objective")
    rospy.logwarn(f"Objective: {objective}")
    if objective == "ContextualForrester":
        rostest.rosrun(
            "bayesopt4ros", "test_python_client", ContextualClientTestCaseForrester
        )
    else:
        raise ValueError("Not a known objective function.")
