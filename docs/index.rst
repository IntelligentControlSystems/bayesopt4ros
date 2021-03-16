BayesOpt4ROS
============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

A Bayesian optimization package for the `Robot Operating System (ROS) <https://www.ros.org>`_.
   

Why BayesOpt4ROS
----------------

.. todo:: Explain motivation behind this package


Introduction to Bayesian Optimization
-------------------------------------

.. todo:: Short introduction to Bayesian Optimization

Documentation
=============

The Bayesian Optimization Service Class
---------------------------------------

.. automodule:: bayesopt_service
.. autoclass:: BayesOptService
   :members: __init__, handler, run

The Bayesian Optimizer
----------------------

.. automodule:: bayesopt4ros.bayesopt
.. autoclass:: BayesianOptimization
   :members: __init__, from_file, next, update_last_y, n_data, y_best, x_best

Acquisition Functions
---------------------

.. automodule:: bayesopt4ros.acq_func
.. autoclass:: UpperConfidenceBound
    :members: __init__, __call__

Optimization
------------

.. automodule:: bayesopt4ros.optim
    :members: minimize_restarts

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
