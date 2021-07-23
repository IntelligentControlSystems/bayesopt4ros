#!/usr/bin/env python3

import rospy
from bayesopt4ros import ContextualBayesOptServer

if __name__ == "__main__":
    try:
        config_name = [p for p in rospy.get_param_names() if "bayesopt_config" in p]
        config_file = rospy.get_param(config_name[0])
        node = ContextualBayesOptServer(config_file=config_file)
        node.run()
    except KeyError:
        rospy.logerr("Could not find the config file.")
    except rospy.ROSInterruptException:
        pass
