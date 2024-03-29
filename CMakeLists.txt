cmake_minimum_required(VERSION 3.0.2)
project(bayesopt4ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  roscpp
  rospy
  std_msgs
  message_generation
  roslaunch
  rostest
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate actions in the 'action' folder
add_action_files(
  DIRECTORY action
  FILES 
  BayesOpt.action
  BayesOptState.action
  ContextualBayesOpt.action
  ContextualBayesOptState.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  actionlib_msgs
  std_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
 CATKIN_DEPENDS actionlib actionlib_msgs roscpp rospy std_msgs message_runtime
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

catkin_install_python(PROGRAMS
  nodes/bayesopt_node.py
  nodes/contextual_bayesopt_node.py
  test/integration/test_client_python.py
  test/integration/test_client_contextual_python.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Testing ##
#############

# Add gtest based cpp test target and link libraries
add_rostest_gtest(test_client_cpp test/integration/test_client_cpp.test test/integration/test_client_cpp.cpp)
target_link_libraries(test_client_cpp ${catkin_LIBRARIES})

# Add folders to be run by python
# Note: I have not found a way to make this more automatic.

# Upper Confidence Bound 
add_rostest(test/integration/test_client_python.test ARGS
  objective:=Forrester
  bayesopt_config:=${PROJECT_SOURCE_DIR}/test/integration/configs/forrester_ucb.yaml)
add_rostest(test/integration/test_client_python.test ARGS
  objective:=ThreeHumpCamel
  bayesopt_config:=${PROJECT_SOURCE_DIR}/test/integration/configs/three_hump_camel_ucb.yaml)

# Expected Improvement
add_rostest(test/integration/test_client_python.test ARGS
  objective:=Forrester
  bayesopt_config:=${PROJECT_SOURCE_DIR}/test/integration/configs/forrester_ei.yaml)
add_rostest(test/integration/test_client_python.test ARGS
  objective:=ThreeHumpCamel
  bayesopt_config:=${PROJECT_SOURCE_DIR}/test/integration/configs/three_hump_camel_ei.yaml)

# Negating the objective
add_rostest(test/integration/test_client_python.test ARGS
  objective:=NegativeForrester
  bayesopt_config:=${PROJECT_SOURCE_DIR}/test/integration/configs/neg_forrester_ucb.yaml)

# Noisy objective with UCB
add_rostest(test/integration/test_client_python.test ARGS
  objective:=NoisyForrester
  bayesopt_config:=${PROJECT_SOURCE_DIR}/test/integration/configs/noisy_forrester_ucb.yaml)

# Contextual setting
add_rostest(test/integration/test_client_contextual_python.test ARGS
  objective:=ContextualForrester
  bayesopt_config:=${PROJECT_SOURCE_DIR}/test/integration/configs/contextual_forrester_ucb.yaml)
