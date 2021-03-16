BayesOpt4ROS
============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

A Bayesian optimization package for the `Robot Operating System (ROS) <https://www.ros.org>`_ developed by the `Intelligent Control Systems (ICS) <https://idsc.ethz.ch/research-zeilinger.html>`_ group at ETH Zurich.

.. warning:: This project is under active development.
   Therefore you might experience breaking changes without any warnings (sorry in advance).
   

Why BayesOpt4ROS
----------------

.. todo:: Explain motivation behind this package


Introduction to Bayesian Optimization
-------------------------------------

.. todo:: Short introduction to Bayesian Optimization


How to Use BayesOpt4ROS
------------------------

The following instructions will help you to get started with the BayesOpt4ROS package.
You can directly clone the repository into your ``catkin_workspace/src/`` directory alongside your other packages.
For the sake of this short tutorial, we have set up an exemplary workspace in `this repository <https://github.com/lukasfro/bayesopt4ros_workspace>`_ and for easy reproducibility we are are using Docker.
However, feel free to set up the package to your preferences.
If you do not have `Docker <https://www.docker.com/get-started>`_ installed, now would be a good time to do so.

.. note:: If you are working on a MacBook M1 with ARM chip, you need to adapt the Dockerfile to pull the right ROS image.
    Just have a look `here <https://github.com/lukasfro/bayesopt4ros_workspace/blob/main/Dockerfile>`_ and uncomment the second line.
    The image based on ARM does not contain all the ROS tools that you need for development (e.g., rviz) but is typically used only for production.


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

Documentation generated: |today|