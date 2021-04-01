#!/usr/bin/env python3

import actionlib
import rospy

from functools import wraps
from typing import Callable

from bayesopt4ros import ContextualBayesianOptimization, util
from bayesopt4ros.msg import ContextualBayesOptResult, ContextualBayesOptAction

from .bayesopt_server import BayesOptServer


class ContextualBayesOptServer(BayesOptServer):
    """The contextual Bayesian optimization server node.

    Acts as a layer between the actual contextual Bayesian optimization and ROS.
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
        """The ContextualBayesOptServer class initializer.

        For paramters see :class:`bayesopt_server.BayesOptServer`.
        """
        super().__init__(
            config_file=config_file,
            server_name=server_name,
            log_file=log_file,
            anonymous=anonymous,
            log_level=log_level,
            silent=silent,
            node_rate=node_rate
        )

    def _initialize_bayesopt(self, config_file):
        try:
            self.bo = ContextualBayesianOptimization.from_file(config_file)
        except Exception as e:
            rospy.logerr(f"[BayesOpt] Something went wrong with initialization: '{e}'")
            rospy.signal_shutdown("Initialization of BayesOpt failed.")
        self.result = ContextualBayesOptResult()

    def _initialize_server(self, server_name):
        self.server = actionlib.SimpleActionServer(
            server_name,
            ContextualBayesOptAction,
            execute_cb=self.execute_callback,
            auto_start=False,
        )

    def _print_goal(self, goal):
        # TODO(lukasfro): also print the context -> make utility function for list to string
        if not self.request_count == 1:
            rospy.loginfo(self._log_prefix + f"New value:   {goal.y_new:.3f}")
        else:
            rospy.loginfo(self._log_prefix + f"Discard value:   {goal.y_new:.3f}")
