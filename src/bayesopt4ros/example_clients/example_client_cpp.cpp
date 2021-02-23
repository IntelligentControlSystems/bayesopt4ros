#include "math.h"
#include "ros/ros.h"
#include <unistd.h>

#include "bayesopt4ros/BayesOptSrv.h"


std::string vecToString(const std::vector<double> &v, int precision) {
    /*! Small helper to get a string representation from a numeric vector.

    Inspiration from: https://github.com/bcohen/leatherman/blob/master/src/print.cpp

    @param v            Vector to be converted to a string.
    @param precision    Number of decimal points to which numbers are shown.

    @return String representation of the vector.

    */
    std::stringstream ss;
    ss << "[";
    for(std::size_t i = 0; i < v.size(); ++i) {
        ss << std::fixed << std::setw(precision) << std::setprecision(precision) << std::showpoint << v[i];
        if (i < v.size() - 1) ss << ", ";
    }
    ss << "]";
    return ss.str();
}

double forresterFunction(const std::vector<double>& x) {
    /*! The Forrester test function for global optimization.

    See definition here: https://www.sfu.ca/~ssurjano/forretal08.html

    Note: We multiply by -1 to maximize the function instead of minimizing.

    @param x    Input to the function.

    @return Function value at given inputs.
    */
    double x0 = x[0];
    return -1.0 * (pow(6.0 * x0 - 2.0, 2) * sin(12.0 * x0 - 4.0));
}


int main(int argc, char **argv)
{
    // Set up ROS
    ros::init(argc, argv, "TestClient");
    if (argc != 2) {
        ROS_INFO("Usage: example_client_cpp <objective>  Choices: [Forrester, ]");
    }
    ros::NodeHandle n;

    // Create client node and corresponding service
    ros::ServiceClient node = n.serviceClient<bayesopt4ros::BayesOptSrv>("BayesOpt");
    bayesopt4ros::BayesOptSrv srv;

    // First value is just to trigger the service
    srv.request.value = 0.0;
    bool success = node.call(srv);
    std::size_t try_count = 0;

    // Try several times before giving up and shutting down
    while (!success) {
        ROS_WARN("[Client] No response from service.");
        success = node.call(srv);
        usleep(1000000);  // 1000000 mu sec = 1 second
        try_count++;
        if (try_count >= 5) {
            ROS_WARN("[Client] Server is not responding after several tries. Exiting.");
            ros::shutdown();
            return 0;
        }
    } 

    // Reading the answer
    std::vector<double> x_new = srv.response.next;

    // Start querying the BayesOpt service until it reached max iterations
    std::size_t iter = 0;
    while (true) {
        ROS_INFO("[Client] Iteration %lu", iter+1);
        std::string result_string = "[Client] x_new = " + vecToString(x_new, 3);
        ROS_INFO_STREAM(result_string);
        
        // Emulate experiment by querying the objective function
        srv.request.value = forresterFunction(x_new);
        ROS_INFO("[Client] y_new = %.2f", srv.request.value);

        // Request service and obtain new parameters
        success = node.call(srv);
        if (success) {
            x_new = srv.response.next;
        } else {
            ROS_WARN("[Client] Invalid response. Shutting down!");
            break;
        }
        iter++;
    }
    ros::shutdown();
    return 0;
}