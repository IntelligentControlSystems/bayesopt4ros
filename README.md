# BayesOpt4ROS

<p align="center">
  <a href="https://github.com/lukasfro/bayesopt4ros/actions">
    <img src="https://github.com/IntelligentControlSystems/bayesopt4ros/actions/workflows/continuous_integration.yml/badge.svg" alt="Integration test status badge">
  </a>
  
  <a href="https://github.com/lukasfro/bayesopt4ros/actions">
    <img src="https://github.com/IntelligentControlSystems/bayesopt4ros/actions/workflows/documentation_deployment.yml/badge.svg" alt="Documentation deployment status badge">
  </a>
  
  <a href="https://zenodo.org/badge/latestdoi/339402959">
    <img src="https://zenodo.org/badge/339402959.svg" alt="DOI">
  </a>
</p>

A Bayesian Optimisation package for ROS developed by the [Intelligent Control Systems (ICS)](https://idsc.ethz.ch/research-zeilinger.html) group at ETH Zurich. 

## Documentation

We provide a official and up-to-date documentation [here](https://intelligentcontrolsystems.github.io/bayesopt4ros/).

## Getting started

For a short tutorial on how to get started with the BayesOpt4ROS package, please see the [official documentation](https://intelligentcontrolsystems.github.io/bayesopt4ros/getting_started.html).

## Contributing

In case you find our package helpful and want to contribute, please either raise an issue or directly make a pull request.

- We follow the [NumPy convention](https://numpydoc.readthedocs.io/en/latest/format.html) for docstrings
- We use the [Black formatter](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html) with the `--line-length` option set to `88`

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
rostest bayesopt4ros test_client_python 
```

## Citing BayesOpt4ROS

If you found our package helpful in your research, we would appreciate if you cite it as follows:
```
@misc {Froehlich2021BayesOpt4ROS, 
  author       = {Fr\"ohlich, Lukas P. and Carron, Andrea},
  title        = {BayesOpt4ROS: A {B}ayesian Optimization package for the Robot Operating System},
  howpublished = {\url{ https://github.com/IntelligentControlSystems/bayesopt4ros}},
  year         = {2021},
  note         = {DOI 10.5281/zenodo.5560966},
}
```
