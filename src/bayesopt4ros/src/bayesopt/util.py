import argparse
import rospy


def service_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-f",
        "--settings_file",
        help="File containing all settings for Bayesian Optimization experiment",
        type=str,
    )

    parser.add_argument(
        "-v",
        "--log_level",
        help="Specify the verbosity of terminal output",
        type=int,
        choices=[rospy.DEBUG, rospy.INFO, rospy.WARN],
        default=rospy.INFO,
    )
    

    return parser
