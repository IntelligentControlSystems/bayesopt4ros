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
    ):
        rospy.init_node(
            self.__class__.__name__, anonymous=anonymous, log_level=log_level
        )
        self.srv = rospy.Service(service_name, BayesOptSrv, self.handler)
        self.request_count = 0
        # Non-existing file -> create new one
        # Existing file -> check if results are consistent (dim, etc.) and start from there
        self.log_file = log_file  # store all results to this file
        self.settings_file = settings_file
        self.bo = BayesianOptimization.from_file(settings_file)

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
        if self.max_iter and self.request_count > self.max_iter:
            rospy.logwarn(
                "[BayesOpt]: Maximum number of iterations reached. Shutting down!"
            )
            rospy.signal_shutdown(
                "Maximum number of iterations reached. Shutting down BayesOpt."
            )
        if self.request_count == 1 and self.verbose:
            rospy.loginfo(
                self._log_prefix
                + f"First request, discarding function value: {req.value}"
            )

        if self.verbose:
            rospy.loginfo(self._log_prefix + "Updating GP model...")
        sleep(0.2)

        if self.verbose:
            rospy.loginfo(self._log_prefix + "Computing next point...")
        sleep(0.8)
        next = list(np.random.rand(self.input_dim))
        if self.verbose:
            p_string = ", ".join([f"{xi:.3f}" for xi in next])
            rospy.loginfo(self._log_prefix + f"Next: [{p_string}]")
        return BayesOptSrvResponse(next)

    @staticmethod
    def run():
        rospy.spin()


if __name__ == "__main__":
    parser = util.service_argparser()
    args = parser.parse_args()
    try:
        node = BayesOptService(
            settings_file=args.settings_file,
            log_level=args.log_level,
        )
        node.run()
    except rospy.ROSInterruptException:
        pass
