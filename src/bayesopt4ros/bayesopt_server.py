#!/usr/bin/env python3

import actionlib
import rospy


from bayesopt4ros import BayesianOptimization, util
from bayesopt4ros.msg import BayesOptResult, BayesOptAction
from bayesopt4ros.msg import BayesOptStateResult, BayesOptStateAction


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

        self._initialize_bayesopt(BayesianOptimization, config_file)
        self._initialize_parameter_server(server_name)
        self._initialize_state_server(server_name + "State")
        self.parameter_server.start()
        self.state_server.start()

        self.request_count = 0
        self.log_file = log_file
        self.config_file = config_file
        self.silent = silent
        self.rosrate = rospy.Rate(node_rate)
        rospy.loginfo(self._log_prefix + "Ready to receive requests.")

    @util.count_requests
    def next_parameter_callback(self, goal: BayesOptAction) -> None:
        """Method that gets called when a new parameter vector is requested.

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
        result = BayesOptResult()
        result.x_new = list(self.bo.next(goal))
        self.parameter_server.set_succeeded(result)
        self._print_result(result) if not self.silent else None

    def state_callback(self, goal) -> None:
        """Method that gets called when the BayesOpt state is requested.

        .. note:: We are calling this `state` instead of `result` to avoid
            confusion with the `result` variable in the action message.

        The action message (goal/result/feedback) is defined here:
        ``action/BayesOptState.action``

        .. literalinclude:: ../action/BayesOptState.action

        Parameters
        ----------
        goal : BayesOptStateAction
            The action (goal) coming from a client.
        """
        state = BayesOptStateResult()

        # Best observed variables
        x_best, y_best = self.bo.get_best_observation()
        state.x_best = list(x_best)
        state.y_best = y_best

        # Posterior mean optimum
        x_opt, f_opt = self.bo.get_optimal_parameters()
        state.x_opt = list(x_opt)
        state.f_opt = f_opt

        self.state_server.set_succeeded(state)

    def _initialize_bayesopt(self, bo_class, config_file):
        try:
            self.bo = bo_class.from_file(config_file)
        except Exception as e:
            rospy.logerr(f"{bo_class.__name__} Something went wrong with initialization: '{e}'")
            rospy.signal_shutdown("Initialization of BayesOpt failed.")

    def _initialize_parameter_server(self, server_name):
        """This server obtains new function values and provides new parameters."""
        self.parameter_server = actionlib.SimpleActionServer(
            server_name,
            BayesOptAction,
            execute_cb=self.next_parameter_callback,
            auto_start=False,
        )

    def _initialize_state_server(self, server_name):
        """This server provides the current state/results of BO."""
        self.state_server = actionlib.SimpleActionServer(
            server_name,
            BayesOptStateAction,
            execute_cb=self.state_callback,
            auto_start=False,
        )

    def _check_final_iter(self, goal):
        if self.bo.max_iter and self.request_count > self.bo.max_iter:
            # Updates model with last function and logs the final GP model
            rospy.logwarn("[BayesOpt] Max iter reached. No longer responding!")
            self.bo.update_last_goal(goal)
            self.parameter_server.set_aborted()
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
        return f"[{self.__class__.__name__}] Iteration {self.request_count}: "

    @staticmethod
    def run() -> None:
        """Simply starts the server."""
        rospy.spin()
