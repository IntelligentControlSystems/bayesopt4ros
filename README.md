# BayesOpt4ROS

<p align="center">
  <a href="https://github.com/lukasfro/bayesopt4ros/actions">
    <img src="https://github.com/lukasfro/bayesopt4ros/actions/workflows/continuous_integration.yml/badge.svg" alt="Integration test status badge">
  </a>
  
  <a href="https://github.com/lukasfro/bayesopt4ros/actions">
    <img src="https://github.com/lukasfro/bayesopt4ros/actions/workflows/documentation_deployment.yml/badge.svg" alt="Documentation deployment status badge">
  </a>
</p>

A Bayesian Optimisation package for ROS developed by the [Intelligent Control Systems (ICS)](https://idsc.ethz.ch/research-zeilinger.html) group at ETH Zurich. 

## Important note about development status

This project is under active development.
Therefore you might experience breaking changes without any warnings (sorry in advance).
As soon as we have a stable version, we will tag the corresponding commit.

## Documentation

We provide a official and up-to-date documentation [here](https://lukasfro.github.io/bayesopt4ros/).

## Getting started

For a short tutorial on how to get started with the BayesOpt4ROS package, please see the [official documentation](https://lukasfro.github.io/bayesopt4ros/getting_started.html).

## Contributing

In case you find our package helpful and want to contribute, please either raise an issue or directly make a pull request.

- We follow the [NumPy convention](https://numpydoc.readthedocs.io/en/latest/format.html) for docstrings

### ToDo List

These are some issues that need to be addressed for the first release of the package.

- [ ] Finish contextual setting
- [ ] Add additional SimpleActionServer that can be queried to get the current state of BO, which is especially helpful at the end of optimization. Here are all the details that are also stored in the log file.
- [ ] Finalize documentation 
- [ ] Create short paper that explains BO in general and this package in particular
- [ ] Noisy EI acquisition function
- [ ] Generally, think about long term strategy for different acquisition functions
- [ ] Figure out why failure cases in tests are not always detected
- [ ] Debug visualization during optimization for 1- and 2-dim. objectives
- [ ] Make code compatible with BoTorch 0.5.0 update
- [ ] Instead of shutting down the full node, tell the client via feedback that the server is no longer available after the maximum number of iterations has been reached
- [ ] Properly distinguish between private/protected/public properties [see here](https://www.tutorialsteacher.com/python/public-private-protected-modifiers)
## Testing

To facilitate easier contribution, we have a continuous integration pipeline set up using Github Actions.
To run the tests locally you can use the following commands:

### Unit tests
```bash
pytest src/bayesopt4ros/test/unit/
```

### Integration tests
To run all integration tests (this command is executed in the CI pipeline):
```bash
catkin_make_isolated -j1 --catkin-make-args run_tests && catkin_test_results
```

Or if you want to just run a specific integration test:
```bash
rostest bayesopt4ros test_client_python ...
```
