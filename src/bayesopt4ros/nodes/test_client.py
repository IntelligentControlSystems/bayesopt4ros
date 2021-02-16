#!/usr/bin/env python

from __future__ import print_function

import sys
import rospy
from bayesopt4ros.srv import BayesOptSrv


class TestClient(object):
    def __init__(self, service_name):
        self.service_name = service_name

    def request(self, value):
        rospy.wait_for_service(self.service_name)
        try:
            bayesopt_request = rospy.ServiceProxy(self.service_name, BayesOptSrv)
            response = bayesopt_request(value)
            return response.next
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")


def usage():
    return "%s [x y]" % sys.argv[0]


if __name__ == "__main__":
    if len(sys.argv) == 2:
        x = float(sys.argv[1])
    else:
        print(usage())
        sys.exit(1)
    # print(f"Requesting {x} + {y}")
    client = TestClient(service_name="BayesOpt")
    next = client.request(x)
    rospy.loginfo(f"[Client] {next}")
