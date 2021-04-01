#!/usr/bin/env python3

import actionlib
import rospy

from functools import wraps
from typing import Callable

from bayesopt4ros import BayesianOptimization, util
from bayesopt4ros.msg import BayesOptResult, BayesOptAction


class BayesOptServer(object):
    """The Bayesian optimization server node.

    Acts as a layer between the actual Bayesian optimization and ROS.
    """

    def __init__(
        self,
        config_file: str,
        server_name: str = "BayesOpt",
        log_file: str = None,
        anonymous: bool = True,
        log_level: int = rospy.INFO,
        silent: bool = False,
        node_rate: float = 5.0,
    ) -> None:
        """The BayesOptServer class initializer.

        Parameters
        ----------
        config_file : str
            File that describes all settings for Bayesian optimization.
        server_name : str
            Name of the server that is used for ROS.
        log_file : str
            All input/output pairs are logged to this file.
        anonymous : bool
            Flag if the node should be anonymous or not (see ROS documentation).
        log_level : int
            Controls the log_level of the node's output.
        silent : bool
            Controls the verbosity of the node's output.
        node_rate : float
            Rate at which the server gives feedback.
        """
        rospy.init_node(
            self.__class__.__name__,
            anonymous=anonymous,
            log_level=log_level,
        )

        self._initialize_server(server_name)
        self.server.start()
        self._initialize_bayesopt(config_file)

        self.request_count = 0
        self.log_file = log_file
        self.config_file = config_file
        self.silent = silent
        self.rosrate = rospy.Rate(node_rate)
        rospy.loginfo(self._log_prefix + "Ready to receive requests.")

    def count_requests(func: Callable) -> Callable:
        """Decorator that keeps track of number of requests.

        Parameters
        ----------
        func : Callable
            The function to be decorated.

        Returns
        -------
        Callable
            The decorated function.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.request_count += 1
            ret_val = func(self, *args, **kwargs)
            return ret_val

        return wrapper

    @count_requests
    def execute_callback(self, goal) -> None:
        """Function that acts upon an action coming from a client.

        When the BayesOpt iteration is finished, the server internal state is
        set to 'succeeded' such that the respective ``done_callback()`` method
        of the client is called.

        The action message (goal/result/feedback) is defined here:
        ``action/BayesOpt.action``

        .. literalinclude:: ../action/BayesOpt.action

        Parameters
        ----------
        goal : BayesOptAction
            The action (goal) coming from a client.
        """

        self._print_goal(goal) if not self.silent else None
        if self._check_final_iter(goal):
            return  # Do not continue once we reached maximum iterations

        # Obtain the new parameter values.
        # (ROS messages can only deal with list, not torch.tensor)
        self.result.x_new = list(self.bo.next(goal))
        self.server.set_succeeded(self.result)
        self._print_result(self.result) if not self.silent else None

    def _initialize_bayesopt(self, config_file):
        try:
            self.bo = BayesianOptimization.from_file(config_file)
        except Exception as e:
            rospy.logerr(f"[BayesOpt] Something went wrong with initialization: '{e}'")
            rospy.signal_shutdown("Initialization of BayesOpt failed.")
        self.result = BayesOptResult()

    def _initialize_server(self, server_name):
        self.server = actionlib.SimpleActionServer(
            server_name,
            BayesOptAction,
            execute_cb=self.execute_callback,
            auto_start=False,
        )

    def _check_final_iter(self, goal):
        if self.bo.max_iter and self.request_count > self.bo.max_iter:
            # Updates model with last function and logs the final GP model
            rospy.logwarn("[BayesOpt] Max iter reached. Shutting down!")
            self.bo.update_last_goal(goal)
            self.server.set_aborted()
            rospy.signal_shutdown("Maximum number of iterations reached")
            return True
        else:
            return False
            
    def _print_goal(self, goal):
        if not self.request_count == 1:
            rospy.loginfo(self._log_prefix + f"New value: {goal.y_new:.3f}")
        else:
            rospy.loginfo(self._log_prefix + f"Discard value: {goal.y_new:.3f}")

    def _print_result(self, result):
        s = ", ".join([f"{xi:.3f}" for xi in result.x_new])
        rospy.loginfo(self._log_prefix + f"x_new: [{s}]")
        if self.request_count < self.bo.max_iter:
            rospy.loginfo(self._log_prefix + "Waiting for new request...")

    @property
    def _log_prefix(self) -> str:
        """Convenience property that pre-fixes the logging strings. """
        return f"[BayesOpt] Iteration {self.request_count}: "


    @staticmethod
    def run() -> None:
        """Simply starts the server."""
        rospy.spin()


if __name__ == "__main__":
    # TODO(lukasfro): use rospy.getparam()
    parser = util.server_argparser()
    args, unknown = parser.parse_known_args()
    try:
        server = BayesOptServer(
            config_file=args.config_file,
            log_level=args.log_level,
            silent=args.silent,
        )
        server.run()
    except rospy.ROSInterruptException:
        pass
