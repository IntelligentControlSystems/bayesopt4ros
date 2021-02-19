#!/usr/bin/env python

import rospy
from bayesopt4ros.srv import BayesOptSrv

from bayesopt.objectives import quadratic


class TestClient(object):
    def __init__(self, service_name):
        rospy.init_node(self.__class__.__name__, anonymous=True, log_level=rospy.INFO)
        self.service_name = service_name

    def request(self, value):
        rospy.wait_for_service(self.service_name)
        try:
            bayesopt_request = rospy.ServiceProxy(self.service_name, BayesOptSrv)
            response = bayesopt_request(value)
            return response.next
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")

    def run(self):
        x_new = self.request(0.0)  # First value is just to trigger the service
        for iter in range(16):
            rospy.loginfo(f"[Client] Iteration {iter + 1}")
            if x_new is not None:
                p_string = ", ".join([f"{xi:.3f}" for xi in x_new])
                rospy.loginfo(f"[Client] x_new = [{p_string}]")
                y_new = quadratic(x_new, dim=2)
                rospy.loginfo(f"[Client] y_new = {y_new:.2f}")
            else:
                rospy.logwarn("[Client] Invalid response. Shutting down!")
                rospy.signal_shutdown("Invalid response from BayesOptService.")
            x_new = self.request(y_new)


if __name__ == "__main__":
    try:
        node = TestClient(service_name="BayesOpt")
        node.run()
    except rospy.ROSInterruptException:
        pass
