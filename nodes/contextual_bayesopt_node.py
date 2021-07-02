#!/usr/bin/env python3

import rospy
from bayesopt4ros import ContextualBayesOptServer

if __name__ == "__main__":
    try:
        config_file = rospy.get_param("/bayesopt_config")
        node = ContextualBayesOptServer(config_file=config_file)
        node.run()
    except KeyError:
        rospy.logerr("Could not find the config file.")
    except rospy.ROSInterruptException:
        pass
