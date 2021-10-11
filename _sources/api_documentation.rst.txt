API Documentation
=================

The Server Classes
------------------

.. automodule:: bayesopt_server
.. autoclass:: BayesOptServer
   :members: __init__, next_parameter_callback, state_callback, run

.. automodule:: contextual_bayesopt_server
.. autoclass:: ContextualBayesOptServer
   :members: __init__, next_parameter_callback, state_callback, run


The Optimizer Classes
---------------------

.. automodule:: bayesopt4ros.bayesopt
.. autoclass:: BayesianOptimization
   :members: __init__, from_file, next, update_last_goal, get_optimal_parameters, get_best_observation, constant_config_parameters

.. automodule:: bayesopt4ros.contextual_bayesopt
.. autoclass:: ContextualBayesianOptimization
   :members: __init__, from_file, next, update_last_goal, get_optimal_parameters, get_best_observation, constant_config_parameters


Utilities
---------

.. automodule:: bayesopt4ros.data_handler
.. autoclass:: DataHandler
   :members: __init__, from_file, get_xy, set_xy, add_xy, n_data, x_best, y_best, x_best_accumulate, y_best_accumulate

.. automodule:: bayesopt4ros.util
   :members: count_requests, iter_to_string, create_log_dir
.. autoclass:: PosteriorMean
   :members: __init__, forward

Examplary Clients
-----------------

.. automodule:: test_client_python
.. autoclass:: ExampleClient
   :members: __init__, request_parameter, request_bayesopt_state, run

.. automodule:: test_client_contextual_python
.. autoclass:: ExampleContextualClient
   :members: __init__, request_parameter, request_bayesopt_state, run, sample_context
