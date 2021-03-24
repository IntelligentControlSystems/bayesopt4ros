# BayesOpt4ROS

<p align="center">
  <a href="https://github.com/lukasfro/bayesopt4ros/actions">
    <img src="https://github.com/lukasfro/bayesopt4ros/actions/workflows/continuous_integration.yml/badge.svg" alt="Integration test status badge">
  </a>
  
  <a href="https://github.com/lukasfro/bayesopt4ros/actions">
    <img src="https://github.com/lukasfro/bayesopt4ros/actions/workflows/pages_deployment.yml/badge.svg" alt="Documentation deployment status badge">
  </a>
</p>

A Bayesian Optimisation package for ROS developed by the [Intelligent Control Systems (ICS)](https://idsc.ethz.ch/research-zeilinger.html) group at ETH Zurich. 

## Important note about development status

This project is under active development.
Therefore you might experience breaking changes without any warnings (sorry in advance).
As soon as we have a stable version, we will tag the corresponding commit.

## ToDo List

- [x] [CI] Set up some basic unit tests
- [x] [CI] Set up some basic integration tests
- [x] [CI] Set up basic GitHub actions
- [ ] [BO] Implement expected improvement acqusition function
- [ ] [BO] Implement max-value entropy search acquisition function
- [ ] [BO] Scale input space to unit (hyper)cube
- [ ] [BO] Scale output space to zero mean / unit variance
- [ ] [BO] Set up reasonable hyper priors for GP parameters

## Instructions

The following instructions will help you to get started with the BayesOpt4ROS package.
Throughout the next steps, we assume that the root directory of this repository corresponds to you catkin workspace.
For sake of reproducibility, we are using Docker.
However, feel free to set up the package to your preferences.
If you do not have [Docker](https://www.docker.com/get-started) installed, now would be a good time to do so.

**Note: If you are working on a MacBook M1 with ARM chip, you need to adapt the Dockerfile to pull the right ROS image. Just have a look [here](Dockerfile) and uncomment the second line.**

### Before you start

**Python and ROS:**
The following steps are only required if you are not using the provided Docker image or if you want to integrate BayesOpt4ROS into your own workspace.
Unfortunately, there is no easy way of automating the following steps.

- BayesOpt4ROS uses features from Python3. 
  If you do not have Python3 installed, please do so now.
- Additionally, we require some 3rd party packages (although we try to keep the dependencies to other packages as small as possible).
  All required packages are specified [here](requirements.txt) and can easily be installed via `python3 -m pip install -r requirements.txt`

### Setup & build

Ok, let's get started by building the images for the first time and runnig the container:
```bash
docker-compose build [--no-cache]
docker-compose up
docker-compose run bayesopt4ros
```

Inside the container, build the workspace (sourcing is done for you) via
```bash
. make_source.sh
```

Note the space between the dot and name of the shell script.
If you just type `./make_source.sh`, sourcing of the `devel_isolated/setup.bash` will not work.
This is because when you execute a shell script, a separate process is spawned which is killed upon completion of the script.
If you want to execute the script within the current running process, you'll need to do this explicitly.

Start roscore (I prefer a detached process  - note the trailing `&` - such that I don't have a terminal tab just for roscore):
```bash
roscore &
```

### Starting the BayesOpt service

Now, let's start the BayesOpt service with a examplary [configuration file](src/bayesopt4ros/configs/example_config_forrrester.yaml).
In this configuration file, we give the service some basic information about the expriment that we want to run, such as the dimensionality of the optimization variable, the search space, etc.
```bash
rosrun bayesopt4ros bayesopt_service.py -f src/bayesopt4ros/configs/example_config_forrrester.yaml
```

You now should see something similar to 
```bash
[INFO] [1613488820.444498]: [BayesOpt] Iteration 0: Ready to receive requests
```

The service is now ready to be contacted by any client node.
The communication protocol is defined via the [BayesOptSrv](src/bayesopt4ros/srv/BayesOptSrv.srv): the client sends a `float64` to the service, which in turn responds with a `list` (Python) / `std::vector` (C++) of `float64`.

### Starting an exemplary client

The client code would typically be implemented by you as it very much depends on your application/experiment.
We implemented to test clients (Python and C++) such that you can see what you need to do for your projects.
For this examples, our goal is to find the global optimum of the [Forrester function](https://www.sfu.ca/~ssurjano/forretal08.html).
**Note that BayesOpt4ROS assumes that you want to maximize your objective. Hence, we multiply the Forrester function with -1.**

Open up another shell and attach to the current container via
```bash
docker exec -it <container_id> bash
```
where `<container_id>` is the ID of your running container (find it via `docker ps`). 

You are now attached to the same container that is already running (you can see this via the beginning of command line `root@<container_id>`. This should be identical for both your shells).
You do not need to worry about sourcing the `devel_isolated/setup.bash` script again, since we added this to the `.bashrc` in the Dockerfile such that you are good to go.

#### Python client

To start the [Python client](src/bayesopt4ros/example_clients/example_client_python.py), use the following command:
```bash
rosrun bayesopt4ros example_client_python.py --objective Forrester
```

#### C++ client

To start the [C++ client](src/bayesopt4ros/example_clients/example_client_cpp.cpp), use the following command:

```bash
rosrun bayesopt4ros example_client_cpp Forrester
```

No matter what clienet you start, the output from the BayesOpt service should look similar to this:

```
[INFO] [1613749625.157782]: [BayesOpt] Iteration 0: Ready to receive requests
[INFO] [1613749627.303790]: [BayesOpt] Iteration 1: First request, discarding function value: 0.0
[INFO] [1613749627.304340]: [BayesOpt] Iteration 1: Computing next point...
[INFO] [1613749627.304738]: [BayesOpt] Iteration 1: x_new: [0.500]
[INFO] [1613749627.305078]: [BayesOpt]: Waiting for new request...
[INFO] [1613749627.328943]: [BayesOpt] Iteration 2: Value from previous iteration: -0.909
[INFO] [1613749627.329605]: [BayesOpt] Iteration 2: Computing next point...
[INFO] [1613749627.387001]: [BayesOpt] Iteration 2: x_new: [1.000]
[INFO] [1613749627.387782]: [BayesOpt]: Waiting for new request...
...
[INFO] [1613749632.810683]: [BayesOpt] Iteration 15: Value from previous iteration: 6.021
[INFO] [1613749632.811740]: [BayesOpt] Iteration 15: Computing next point...
[INFO] [1613749633.471213]: [BayesOpt] Iteration 15: x_new: [0.757]
[WARN] [1613749633.487271]: [BayesOpt] Max iter reached. Shutting down!
```

Note that the service node shuts down after `max_iter` requests (this can be changed in the configuration file. Choose `max_iter: 0` to let the service run indefinitely).

### Using Roslaunch

For ease of use, you can also start the service and client with a single command:
```bash
roslaunch bayesopt4ros bayesopt_service.launch
```

### Visualizing the result

If you specify a logging directory in the configuration file, the BayesOpt service will store the evaluated points, the current GP model as well as the configuration itself to the logging directory.
We provide a simple plotting script that visualizes the evaluated points as well as the evolution of the best function value and the final GP model.

Assuming that the logging directory is set to `./logs/`, you can visualize the results simple by

```bash
python src/visualize.py -d logs
```

Which should pop up a window similar to this one:

![alt text](doc/readme_example_visualization.png)

On the left, we see function values of all evaluated points (black dots) as well as the evolution of the best function value (red line) as a function the of iterations.
On the right, the final GP model (blue) with all the training data (black dots), as well as the best point (yello star) are visualized.

## Testing

To facilitate easier contribution, we have a continuous integration pipeline set up using Github Actions.
To run the tests locally you can use the following commands:

### Unit tests
```bash
pytest src/bayesopt4ros/test/
```

### Integration tests
```bash
# TODO
```
