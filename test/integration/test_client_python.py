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
from bayesopt4ros.msg import BayesOptStateAction, BayesOptStateGoal, BayesOptStateResult
from bayesopt4ros.msg import ContextualBayesOptAction, ContextualBayesOptGoal


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


class ContextualForrester(Forrester):
    """Inspired by 'Modifications and Alternative Forms' section on the Forrester
    function (https://www.sfu.ca/~ssurjano/forretal08.html), this class extends
    the standard Forrester function by a linear trend with varying slope. The
    slope defines the context and is between 0 (original) and 20.

    See the location and value of the optimum for different contexts below.

    c =  0.0: g_opt = -6.021, x_opt = 0.757
    c =  2.0: g_opt = -5.508, x_opt = 0.755
    c =  4.0: g_opt = -4.999, x_opt = 0.753
    c =  6.0: g_opt = -4.494, x_opt = 0.751
    c =  8.0: g_opt = -3.993, x_opt = 0.749
    c = 10.0: g_opt = -4.705, x_opt = 0.115
    c = 12.0: g_opt = -5.480, x_opt = 0.110
    c = 14.0: g_opt = -6.264, x_opt = 0.106
    c = 16.0: g_opt = -7.057, x_opt = 0.101
    c = 18.0: g_opt = -7.859, x_opt = 0.097
    c = 20.0: g_opt = -8.670, x_opt = 0.092
    """

    dim = 2
    input_dim = 1
    context_dim = 1

    _bounds = [(0.0, 1.0), (0.0, 20.0)]

    # I'll defined for a contextual problem
    _optimal_value = None
    _optimizers = None

    # Define optimizer for specific context values
    _test_contexts = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    _contextual_optimizer = [
        0.757,
        0.755,
        0.753,
        0.751,
        0.749,
        0.115,
        0.110,
        0.106,
        0.101,
        0.097,
        0.092,
    ]
    _contextual_optimal_values = [
        -6.021,
        -5.508,
        -4.999,
        -4.494,
        -3.993,
        -4.705,
        -5.480,
        -6.264,
        -7.057,
        -7.859,
        -8.670,
    ]

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return super().evaluate_true(X[:, 0]) + X[:, 1] * (X[:, 0] - 0.5)


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
        if objective == "Forrester":
            self.func = Forrester()
        elif objective == "NegativeForrester":
            self.func = Forrester(negate=True)
        elif objective == "ThreeHumpCamel":
            self.func = ShiftedThreeHumpCamel()
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


class ExampleContextualClient(object):
    """A demonstration on how to use the contexutal BayesOpt server from a Python node."""

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
            self.func = ContextualForrester()
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
            An array containing the new parameters suggested by contextual BayesOpt server.
        """
        goal = ContextualBayesOptGoal(y_new=y_new, c_new=c_new)
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        return torch.tensor(result.x_new)

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


class ClientTestCase(unittest.TestCase):
    """Integration test cases for exemplary Python client."""

    _objective_name = None
    _maximize = True

    def test_objective(self) -> None:
        """Testing the client on the defined objective function. """

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

        print(f"{result.x_best = }, {result.y_best = }")
        print(f"{result.x_opt = }, {result.f_opt = }")

        # Be kind w.r.t. precision of solution
        np.testing.assert_almost_equal(result.x_opt, x_opt, decimal=2)
        np.testing.assert_almost_equal(result.f_opt, f_opt, decimal=2)


class ContextualClientTestCase(unittest.TestCase):
    """Integration test cases for exemplary contextual Python client."""

    _objective_name = None
    _maximize = True

    def test_objective(self) -> None:
        """Testing the client on the defined objective function and couple of contexts."""
        node = ExampleContextualClient(
            server_name="ContextualBayesOpt",
            objective=self._objective_name,
            maximize=self._maximize,
        )
        node.run()

        for c, x, y in zip(
            node.func._test_contexts,
            node.func._contextual_optimizer,
            node.func._contextual_optimal_values,
        ):
            print(c, x, y)



class ClientTestCaseForrester(ClientTestCase):
    _objective_name = "Forrester"
    _maximize = False


class ClientTestCaseNegativeForrester(ClientTestCase):
    _objective_name = "NegativeForrester"
    _maximize = True


class ClientTestCaseThreeHumpCamel(ClientTestCase):
    _objective_name = "ThreeHumpCamel"
    _maximize = False


class ContextualClientTestCaseForrester(ContextualClientTestCase):
    _objective_name = "ContextualForrester"
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
    elif objective == "ThreeHumpCamel":
        rostest.rosrun(
            "bayesopt4ros", "test_python_client", ClientTestCaseThreeHumpCamel
        )
    elif objective == "ContextualForrester":
        rostest.rosrun(
            "bayesopt4ros", "test_python_client", ContextualClientTestCaseForrester
        )
    else:
        raise ValueError("Not a known objective function.")
