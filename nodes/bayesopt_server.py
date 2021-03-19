#!/usr/bin/env python3

import rospy
from bayesopt4ros.msg import BayesOptResult, BayesOptAction


class BayesOptServer(object):
    _result = BayesOptResult

    def __init__(self) -> None:
        rospy.init_node(self.__class__.__name__)
        rospy.loginfo("[Server] Initialization done")

    @staticmethod
    def run() -> None:
        rospy.spin()


if __name__ == "__main__":
    try:
        server = BayesOptServer()
        server.run()
    except rospy.ROSInterruptException:
        pass
