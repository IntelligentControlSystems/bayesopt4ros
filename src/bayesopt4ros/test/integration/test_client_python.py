import unittest
import itertools
import numpy as np
import rospy
import rostest
import sys

from typing import Callable, Union

from bayesopt4ros.srv import BayesOptSrv


def forrester_function(x: Union[np.ndarray, float]) -> np.ndarray:
    """! The Forrester test function for global optimization.

    See definition here: https://www.sfu.ca/~ssurjano/forretal08.html

    Note: We multiply by -1 to maximize the function instead of minimizing.

    @param x    Input to the function.

    @return Function value at given inputs.
    """
    x = np.array(x)
    return -1 * ((6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)).squeeze()


def three_hump_camel_function(x: np.ndarray) -> np.ndarray:
    """! The Three-Hump Camel test function for global optimization.

    See definition here: https://www.sfu.ca/~ssurjano/camel3.html

    Note: We multiply by -1 to maximize the function instead of minimizing.

    @param x    Input to the function.

    @return Function value at given inputs.
    """
    x = np.atleast_2d(x)
    x1_terms = 2 * x[:, 0] ** 2 - 1.05 * x[:, 0] ** 4 + x[:, 0] ** 6 / 6
    x12_terms = x[:, 0] * x[:, 1]
    x2_terms = x[:, 1] ** 2
    return -1 * (x1_terms + x12_terms + x2_terms).squeeze()


class ExampleClient(object):
    """! A demonstration on how to use the BayesOpt service from a Python node. """

    def __init__(self, service_name: str, objective: Callable) -> None:
        """! Initializer of the client that queries the BayesOpt service.

        @param service_name     Name of the service (needs to be consistent with service node).
        @param objective        Name of the example objective.
        """
        rospy.init_node(self.__class__.__name__, anonymous=True, log_level=rospy.INFO)
        self.service_name = service_name
        self.y_best, self.x_best = -np.inf, None
        if objective == "Forrester":
            self.func = forrester_function
        elif objective == "ThreeHumpCamel":
            self.func = three_hump_camel_function
        else:
            raise ValueError("No such objective.")

    def request(self, value: float) -> np.ndarray:
        """! Method that handles interaction with the service node.

        @param value    The function value obtained from the objective/experiment.

        @return An array containing the new parameters suggested by BayesOpt service.
        """
        rospy.wait_for_service(self.service_name)
        try:
            bayesopt_request = rospy.ServiceProxy(self.service_name, BayesOptSrv)
            response = bayesopt_request(value)
            return response.next
        except rospy.ServiceException as e:
            rospy.logwarn("[Client] Invalid response. Shutting down!")
            rospy.signal_shutdown("Invalid response from BayesOptService.")

    def run(self) -> None:
        """! Method that emulates client behavior."""
        # First value is just to trigger the service
        x_new = self.request(0.0)

        # Start querying the BayesOpt service until it reached max iterations
        for iter in itertools.count():
            rospy.loginfo(f"[Client] Iteration {iter + 1}")
            p_string = ", ".join([f"{xi:.3f}" for xi in x_new])
            rospy.loginfo(f"[Client] x_new = [{p_string}]")
            # Emulate experiment by querying the objective function
            y_new = self.func(x_new)

            if y_new > self.y_best:
                self.y_best = y_new
                self.x_best = x_new
            rospy.loginfo(f"[Client] y_new = {y_new:.2f}, y_best = {self.y_best:.2f}")

            # Request service and obtain new parameters
            x_new = self.request(y_new)
            if x_new is None:
                break


class ClientTestCaseForrester(unittest.TestCase):
    """! Integration test cases for exemplary Python client. """

    def test_forrester(self) -> None:
        """! Testing client on 1-dimensional Forrester function."""
        node = ExampleClient(service_name="BayesOpt", objective="Forrester")
        node.run()

        # Be kind w.r.t. precision of solution
        np.testing.assert_almost_equal(node.x_best, np.array([0.757]), decimal=3)
        np.testing.assert_almost_equal(node.y_best, np.array([6.021]), decimal=3)


class ClientTestCaseThreeHumpCamel(unittest.TestCase):
    """! Integration test cases for exemplary Python client. """

    def test_three_hump_camel(self) -> None:
        """! Testing client on 2-dimensional Three-Hump camel function."""
        node = ExampleClient(service_name="BayesOpt", objective="ThreeHumpCamel")
        node.run()

        # Be kind w.r.t. precision of solution
        np.testing.assert_almost_equal(node.x_best, np.array([0.0, 0.0]), decimal=2)
        np.testing.assert_almost_equal(node.y_best, np.array([0.0]), decimal=3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--objective",
        help="Objective's name",
        choices=["forrester", "three_hump_camel"],
    )
    args, unknown = parser.parse_known_args()

    # Note: unfortunately, rostest.rosrun does not allow to parse arguments
    # This can probably be done more efficiently but honestly, the ROS documentation for
    # integration testing is kind of outdated and not very thorough...
    if args.objective == "forrester":
        rostest.rosrun("bayesopt4ros", "test_python_client", ClientTestCaseForrester)
    elif args.objective == "three_hump_camel":
        rostest.rosrun(
            "bayesopt4ros", "test_python_client", ClientTestCaseThreeHumpCamel
        )
    else:
        raise ValueError("Not a known objective function.")
