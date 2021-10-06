import actionlib  # type: ignore
import rospy

from bayesopt4ros import ContextualBayesianOptimization, BayesOptServer, util
from bayesopt4ros.msg import (  # type: ignore
    ContextualBayesOptResult,
    ContextualBayesOptAction,
)
from bayesopt4ros.msg import (  # type: ignore
    ContextualBayesOptStateResult,
    ContextualBayesOptStateAction,
)


class ContextualBayesOptServer(BayesOptServer):
    """The contextual Bayesian optimization server node.

    Acts as a layer between the actual contextual Bayesian optimization and ROS.
    """

    def __init__(
        self,
        config_file: str,
        server_name: str = "ContextualBayesOpt",
        log_file: str = None,
        anonymous: bool = True,
        log_level: int = rospy.INFO,
        silent: bool = False,
        node_rate: float = 5.0,
    ) -> None:
        """The ContextualBayesOptServer class initializer.

        For paramters see :class:`bayesopt_server.BayesOptServer`.
        """
        rospy.logdebug("Initializing Contextual BayesOpt Server")
        super().__init__(
            config_file=config_file,
            server_name=server_name,
            log_file=log_file,
            anonymous=anonymous,
            log_level=log_level,
            silent=silent,
            node_rate=node_rate,
        )

        rospy.logdebug("[ContextualBayesOptServer] Initialization done")

    @util.count_requests
    def next_parameter_callback(self, goal: ContextualBayesOptAction) -> None:
        self._print_goal(goal) if not self.silent else None
        if self._check_final_iter(goal):
            return  # Do not continue once we reached maximum iterations

        # Obtain the new parameter values.
        result = ContextualBayesOptResult()
        result.x_new = list(self.bo.next(goal))
        self.parameter_server.set_succeeded(result)
        self._print_result(result) if not self.silent else None

    def state_callback(self, goal: ContextualBayesOptStateAction) -> None:
        state = ContextualBayesOptStateResult()

        # Best observed variables
        x_best, c_best, y_best = self.bo.get_best_observation()
        state.x_best = list(x_best)
        state.c_best = list(c_best)
        state.y_best = y_best

        # Posterior mean optimum for a given context
        x_opt, f_opt = self.bo.get_optimal_parameters(goal.context)
        state.x_opt = list(x_opt)
        state.f_opt = f_opt

        self.state_server.set_succeeded(state)

    def _initialize_bayesopt(self, config_file):
        try:
            self.bo = ContextualBayesianOptimization.from_file(config_file)
        except Exception as e:
            rospy.logerr(
                f"[ContextualBayesOpt] Something went wrong with initialization: '{e}'"
            )
            rospy.signal_shutdown("Initialization of ContextualBayesOpt failed.")

    def _initialize_parameter_server(self, server_name):
        """This server obtains new function values and provides new parameters."""
        self.parameter_server = actionlib.SimpleActionServer(
            server_name,
            ContextualBayesOptAction,
            execute_cb=self.next_parameter_callback,
            auto_start=False,
        )

    def _initialize_state_server(self, server_name):
        """This server provides the current state/results of BO."""
        self.state_server = actionlib.SimpleActionServer(
            server_name,
            ContextualBayesOptStateAction,
            execute_cb=self.state_callback,
            auto_start=False,
        )

    def _print_goal(self, goal):
        if not self.request_count == 1:
            s = self._log_prefix + f"y_n: {goal.y_new:.3f}"
            s += f", c_(n+1) = {util.iter_to_string(goal.c_new, '.3f')}"
            rospy.loginfo(s)
        else:
            rospy.loginfo(self._log_prefix + f"Discard value: {goal.y_new:.3f}")
