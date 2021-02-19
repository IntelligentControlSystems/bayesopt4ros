#!/usr/bin/env python

import argparse
import numpy as np
import rospy

from time import sleep

from bayesopt4ros.srv import BayesOptSrv, BayesOptSrvResponse
from bayesopt import BayesianOptimization
from bayesopt import util


class BayesOptService(object):
    def __init__(
        self,
        settings_file,
        service_name="BayesOpt",
        log_file=None,
        anonymous=True,
        log_level=rospy.INFO,
        silent=False,
    ):
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

    def count_requests(func):
        def wrapper(self, *args, **kwargs):
            self.request_count += 1
            ret_val = func(self, *args, **kwargs)
            return ret_val

        return wrapper

    @property
    def _log_prefix(self):
        return f"[BayesOpt] Iteration {self.request_count}: "

    @count_requests
    def handler(self, req):
        if self.bo.max_iter and self.request_count > self.bo.max_iter:
            rospy.logwarn("[BayesOpt] Max iter reached. Shutting down!")
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
        sleep(0.8)

        x_new = self.bo.next(req.value)
        x_new = list(x_new)  # ROS service is specified as a list

        if not self.silent:
            p_string = ", ".join([f"{xi:.3f}" for xi in x_new])
            rospy.loginfo(self._log_prefix + f"x_new: [{p_string}]")
            if self.request_count < self.bo.max_iter:
                rospy.loginfo("[BayesOpt]: Waiting for new request...")

        return BayesOptSrvResponse(x_new)

    @staticmethod
    def run():
        rospy.spin()


if __name__ == "__main__":
    parser = util.service_argparser()
    args = parser.parse_args()
    try:
        node = BayesOptService(
            config_file=args.config_file,
            log_level=args.log_level,
            silent=args.silent,
        )
        node.run()
    except rospy.ROSInterruptException:
        pass
