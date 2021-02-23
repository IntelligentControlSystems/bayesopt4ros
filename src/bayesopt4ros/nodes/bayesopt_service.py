#!/usr/bin/env python3

from typing import Callable
import rospy
from bayesopt4ros.srv import BayesOptSrv, BayesOptSrvResponse
from bayesopt4ros import BayesianOptimization, util


class BayesOptService(object):
    """! The Bayesian opitmization service node.

    Acts as a layer between the actual Bayesian optimization and ROS.

    Note: We assume that the objective function is to be maximized!
    """

    def __init__(
        self,
        config_file: str,
        service_name: str = "BayesOpt",
        log_file: str = None,
        anonymous: bool = True,
        log_level: int = rospy.INFO,
        silent: bool = False,
    ) -> None:
        """! The BayesOptService class initializer.

        @param config_file      File that describes all settings for Bayesian optimization.
        @param service_name     Name of the service that is used for ROS.
        @log_file               All input/output pairs are logged to this file.
        @anonymous              Flag if the node should be anonymous or not (see ROS documentation).
        @log_level              Controls the log_level of the node's output.
        @silent                 Controls the verbosity of the node's output.
        """
        rospy.init_node(
            self.__class__.__name__, anonymous=anonymous, log_level=log_level
        )
        self.srv = rospy.Service(service_name, BayesOptSrv, self.handler)
        self.request_count = 0
        self.log_file = log_file
        self.config_file = config_file
        self.bo = BayesianOptimization.from_file(config_file)
        self.silent = silent

        rospy.loginfo(self._log_prefix + "Ready to receive requests")

    def count_requests(func: Callable) -> Callable:
        """! Decorator that keeps track of number of requests.

        @param func     The function to be decorated.

        @param The decorated function.
        """

        def wrapper(self, *args, **kwargs):
            self.request_count += 1
            ret_val = func(self, *args, **kwargs)
            return ret_val

        return wrapper

    @property
    def _log_prefix(self) -> str:
        """! Convenience property that pre-fixes the logging strings. """
        return f"[BayesOpt] Iteration {self.request_count}: "

    @count_requests
    def handler(self, req: BayesOptSrv) -> BayesOptSrvResponse:
        """! Function that acts upon the request coming from a client.

        @param req  The request coming from a client. Definition is in /srv/BayesOptSrv.srv

        @returns The corresponding response to the request. Definition is in /srv/BayesOptSrv.srv
        """
        if self.bo.max_iter and self.request_count > self.bo.max_iter:
            # Updates model with last function and logs the final GP model
            rospy.logwarn("[BayesOpt] Max iter reached. Shutting down!")
            self.bo.update_last_y(req.value)
            rospy.signal_shutdown("Maximum number of iterations reached")

        if not self.silent:
            if self.request_count == 1:
                rospy.loginfo(
                    self._log_prefix
                    + f"First request, discarding function value: {req.value}"
                )
            else:
                rospy.loginfo(
                    self._log_prefix + f"Value from previous iteration: {req.value:.3f}"
                )
            rospy.loginfo(self._log_prefix + "Computing next point...")

        # This line actually gets the new parameter values.
        x_new = self.bo.next(req.value)
        # ROS service is specified as a list
        x_new = list(x_new)

        # Pretty-log the response to std out
        if not self.silent:
            s = ", ".join([f"{xi:.3f}" for xi in x_new])
            rospy.loginfo(self._log_prefix + f"x_new: [{s}]")
            if self.request_count < self.bo.max_iter:
                rospy.loginfo("[BayesOpt]: Waiting for new request...")

        return BayesOptSrvResponse(x_new)

    @staticmethod
    def run() -> None:
        """! Method that starts the node. """
        rospy.spin()


if __name__ == "__main__":
    parser = util.service_argparser()
    args, unknown = parser.parse_known_args()
    try:
        node = BayesOptService(
            config_file=args.config_file,
            log_level=args.log_level,
            silent=args.silent,
        )
        node.run()
    except rospy.ROSInterruptException:
        pass
