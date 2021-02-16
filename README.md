# BayesOpt4ROS

[![CircleCI](https://circleci.com/gh/lukasfro/bayesopt4ros.svg?style=shield&circle-token=455400e23bd646c26570706fcdae8b01c3d3611f)]([<>](https://app.circleci.com/pipelines/github/lukasfro/bayesopt4ros))

A Bayesian Optimisation package for ROS. 

## ToDo List

Immediate:
- [ ] [CI] Set up pipeline in circle CI using docker
- [ ] [BO] Create BO object from settings file
- [ ] [BO] Actually implement some BO functionality (UCB first)
- [ ] [CI] Set up some basic unit tests
- [ ] [CI] Set up some basic integration tests
- [ ] [CRS] Figure out how to integrate this with the CRS (or other code bases)

Long-/midterm:
- [ ] [OS] Make this repository public on Github
- [ ] [OS] Make this repository be listed on ROS Index
- [ ] [OS] Using type hinting throughout the project
- [ ] [OS] Proper documentation of all functionality
- [ ] [OS] Write a proper README with instructions, etc. 
- [ ] [BO] Implement contextual BO
- [ ] [BO] Integrate CRBO


## Instructions

In order to get the BayesOpt service and a test client running, follow the steps below.

**Note: the current steup only works for MacBooks with ARM architecture (M1).Further, the development of this package is in alpha stage. If you have an Intel CPU (which you probably have), just exchange the first line of the Dockerfile by `FROM osrf/ros:noetic`. Expect breaking changes at any time!**

```bash
docker-compose build [--no-cache]
docker-compose up
docker-compose run bayesopt4ros
```

Inside the container, build the workspace (sourcing is done for you) via
```bash
. make_source.sh
```
Note the space between the dot and name of the shell script. If you just type `./make_source.sh`, sourcing of the `devel_isolated/setup.bash` will not work. This is because when you execute a shell script, a separate process is spawned which is killed upon completion of the script. If you want to execute the script within the current running process, you'll need to do this explicitly.

Start roscore (I prefer a detached process as I don't need it to be in a separate tab of my terminal):
```bash
roscore &
```

Now, let's start the BayesOpt service:
```bash
rosrun bayesopt4ros bayesopt_service.py
```

You now should see something similar to 
```bash
[INFO] [1613488820.444498]: [BayesOpt] Iteration 0/3: Ready to receive requests
```

Open up another shell and attach to the current container via
```bash
docker exec -it <container_id> bash
```
where `<container_id>` is the ID of your running container (find it via `docker ps`). 

You are now attached to the same container that is already running (you can see this via the beginning of command line `root@<container_id>`. This should be identical for both your shells).
You do not need to worry about sourcing the `devel_isolated/setup.bash` script again, since we added this to the `.bashrc` in the Dockerfile such that you are good to go.
No we execute a request for the BayesOpt service via the test client:
```bash
rosrun bayesopt4ros test_client.py 1.0
```
This command sends the value `1.0` to the service. As we have not yet interacted with the service yet, the first function value is discarded. The client will now receive a tuple of numbers, which corresponds to the new set of parameters that the BayesOpt service is suggesting.
We can run the test client again to obtain a new set of points (and continue to do so for how long you like). I you started the BayesOpt service with a maximum number of iterations (via the `max_iter` argument), the service will shut down after you have sent `max_iter` requests. 

After requesting the service for a couple of times, your output should look similar to
```
[INFO] [1613488820.444498]: [BayesOpt] Iteration 0/3: Ready to receive requests
[INFO] [1613489297.683983]: [BayesOpt] Iteration 1/3: First request, discarding function value: 1.0
[INFO] [1613489297.694952]: [BayesOpt] Iteration 1/3: Updating GP model...
[INFO] [1613489297.898034]: [BayesOpt] Iteration 1/3: Computing next point...
[INFO] [1613489298.722249]: [BayesOpt] Iteration 1/3: Next: [0.506, 0.990, 0.941, 0.023, 0.604]
[INFO] [1613489730.134280]: [BayesOpt] Iteration 2/3: Updating GP model...
[INFO] [1613489730.340919]: [BayesOpt] Iteration 2/3: Computing next point...
[INFO] [1613489731.152201]: [BayesOpt] Iteration 2/3: Next: [0.247, 0.916, 0.019, 0.253, 0.278]
[INFO] [1613489737.693475]: [BayesOpt] Iteration 3/3: Updating GP model...
[INFO] [1613489737.899950]: [BayesOpt] Iteration 3/3: Computing next point...
[INFO] [1613489738.711191]: [BayesOpt] Iteration 3/3: Next: [0.935, 0.391, 0.792, 0.218, 0.323]
[WARN] [1613489741.913561]: [BayesOpt]: Maximum number of iterations reached. Shutting down!
```

**Note: Currently the BayesOpt service just returns random points no actual 'intelligent' behaviour is implemented. The state as of now just serves as a skeleton for further development.**

