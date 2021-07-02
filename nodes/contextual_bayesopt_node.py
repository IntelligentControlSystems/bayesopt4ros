#!/usr/bin/env python3

import rospy
from bayesopt4ros import ContextualBayesOptServer, util

if __name__ == "__main__":
    # TODO(lukasfro): use rospy.getparam()
    parser = util.server_argparser()
    args, unknown = parser.parse_known_args()
    try:
        node = ContextualBayesOptServer(
            config_file=args.config_file,
            log_level=args.log_level,
            silent=args.silent,
        )
        node.run()
    except rospy.ROSInterruptException:
        pass
