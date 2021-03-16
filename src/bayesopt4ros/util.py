import argparse
import rospy


def service_argparser():
    """Sets up the argument parser used for the BayesOpt service."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--config_file",
        help="File containing the configuration for the Bayesian Optimization service",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--log_level",
        help="Specify the verbosity of terminal output",
        type=int,
        choices=[rospy.DEBUG, rospy.INFO, rospy.WARN],
        default=rospy.INFO,
    )

    parser.add_argument(
        "--silent",
        help="Prevents printing status/updates for the service",
        action="store_true",
    )

    return parser
