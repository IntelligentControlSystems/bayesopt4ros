import torch

from botorch.test_functions import SyntheticTestFunction, ThreeHumpCamel


class Forrester(SyntheticTestFunction):
    """The Forrester test function for global optimization.
    See definition here: https://www.sfu.ca/~ssurjano/forretal08.html
    """

    dim = 1
    _bounds = [(0.0, 1.0)]
    _optimal_value = -6.021
    _optimizers = [(0.757)]

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return (6.0 * X - 2.0) ** 2 * torch.sin(12.0 * X - 4.0)


class ShiftedThreeHumpCamel(ThreeHumpCamel):
    """The Three-Hump Camel test function for global optimization.

    See definition here: https://www.sfu.ca/~ssurjano/camel3.html

    .. note:: We shift the inputs as the initial design would hit the optimum directly.
    """

    _optimizers = [(0.5, 0.5)]

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return super().evaluate_true(X - 0.5)


class ContextualForrester(Forrester):
    """Inspired by 'Modifications and Alternative Forms' section on the Forrester
    function (https://www.sfu.ca/~ssurjano/forretal08.html), this class extends
    the standard Forrester function by a linear trend with varying slope. The
    slope defines the context and is between 0 (original) and 20.

    See the location and value of the optimum for different contexts below.

    c =  0.0: g_opt = -6.021, x_opt = 0.757
    c =  2.0: g_opt = -5.508, x_opt = 0.755
    c =  4.0: g_opt = -4.999, x_opt = 0.753
    c =  6.0: g_opt = -4.494, x_opt = 0.751
    c =  8.0: g_opt = -3.993, x_opt = 0.749
    c = 10.0: g_opt = -4.705, x_opt = 0.115
    c = 12.0: g_opt = -5.480, x_opt = 0.110
    c = 14.0: g_opt = -6.264, x_opt = 0.106
    c = 16.0: g_opt = -7.057, x_opt = 0.101
    c = 18.0: g_opt = -7.859, x_opt = 0.097
    c = 20.0: g_opt = -8.670, x_opt = 0.092
    """

    dim = 2
    input_dim = 1
    context_dim = 1

    _bounds = [(0.0, 1.0), (0.0, 20.0)]

    # I'll defined for a contextual problem
    _optimal_value = None
    _optimizers = None

    # Define optimizer for specific context values
    _test_contexts = [
        [0.0],
        [2.0],
        [4.0],
        [6.0],
        [8.0],
        [10.0],
        [12.0],
        [14.0],
        [16.0],
        [18.0],
        [20.0],
    ]
    _contextual_optimizer = [
        [0.757],
        [0.755],
        [0.753],
        [0.751],
        [0.749],
        [0.115],
        [0.110],
        [0.106],
        [0.101],
        [0.097],
        [0.092],
    ]
    _contextual_optimal_values = [
        -6.021,
        -5.508,
        -4.999,
        -4.494,
        -3.993,
        -4.705,
        -5.480,
        -6.264,
        -7.057,
        -7.859,
        -8.670,
    ]

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return super().evaluate_true(X[:, 0]) + X[:, 1] * (X[:, 0] - 0.5)
