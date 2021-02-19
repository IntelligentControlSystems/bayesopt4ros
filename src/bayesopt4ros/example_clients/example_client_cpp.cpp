#include "ros/ros.h"
#include "bayesopt4ros/AddTwoInts.h"
#include "bayesopt4ros/BayesOptSrv.h"
#include <cstdlib>

std::string getString(const std::vector<double> &v, int precision)
{
  // Inspiration from: https://github.com/bcohen/leatherman/blob/master/src/print.cpp
  std::stringstream ss;
  ss << "[";
  for(std::size_t j = 0; j < v.size(); ++j)
  {
    ss << std::fixed << std::setw(precision) << std::setprecision(precision) << std::showpoint << v[j];
    if (j < v.size() - 1) ss << ", ";
  }
  
  ss << "]";
  return ss.str();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "add_two_ints_client");

  ros::NodeHandle n;
  ros::ServiceClient bo_client = n.serviceClient<bayesopt4ros::BayesOptSrv>("BayesOpt");
  bayesopt4ros::BayesOptSrv bo_srv;

  bo_srv.request.value = 1.0;

  if (bo_client.call(bo_srv))
  {
    ROS_INFO("Received the following");
    std::string result_string = "[Client]: " + getString(bo_srv.response.next, 3);
    ROS_INFO_STREAM(result_string);
  }

  return 0;
}