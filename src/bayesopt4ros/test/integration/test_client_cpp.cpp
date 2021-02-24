#include "math.h"
#include "ros/ros.h"
#include <unistd.h>
#include <gtest/gtest.h>

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


class ExampleClient {
    // ! A demonstration on how to use the BayesOpt service from a C++ node.
    public:
        ExampleClient(std::string service_name, std::string objective) {
            /*! Constructor of the client that queries the BayesOpt service.

            @param service_name     Name of the service (needs to be consistent with service node).
            @param objective        Name of the example objective.
            */
            ros::NodeHandle nh;
            objective_ = objective;
            node_ = nh.serviceClient<bayesopt4ros::BayesOptSrv>(service_name);
        }

        void run() {
            /*! Method that emulates client behavior. */

            // First value is just to trigger the service
            node_.waitForExistence();
            srv_.request.value = 0.0;
            bool success = node_.call(srv_);
            std::vector<double> x_new = srv_.response.next;

            // Start querying the BayesOpt service until it reached max iterations
            std::size_t iter = 0;
            while (true) {
                ROS_INFO("[Client] Iteration %lu", iter+1);
                std::string result_string = "[Client] x_new = " + vecToString(x_new, 3);
                ROS_INFO_STREAM(result_string);
                
                // Emulate experiment by querying the objective function
                if (objective_.compare("Forrester") == 0) {
                    srv_.request.value = forresterFunction(x_new);
                } else {
                    ROS_ERROR("[Client] No such objective: %s", objective_.c_str());
                    break;
                }

                if (srv_.request.value > y_best) {
                    y_best = srv_.request.value;
                    x_best = x_new;
                }
                ROS_INFO("[Client] y_new = %.2f, y_best = %.2f", srv_.request.value, y_best);

                // Request service and obtain new parameters
                success = node_.call(srv_);
                if (success) {
                    x_new = srv_.response.next;
                } else {
                    ROS_WARN("[Client] Invalid response. Shutting down!");
                    break;
                }
                iter++;
            }
        }

        double y_best = std::numeric_limits<double>::min();
        std::vector<double> x_best;
    private:
        ros::ServiceClient node_;
        bayesopt4ros::BayesOptSrv srv_;
        std::string objective_;
};


TEST(ClientTestSuite, testForrester)
{
    // Create client node and corresponding service
    ExampleClient client("BayesOpt", "Forrester");
    client.run();
    ros::shutdown();

    // Be kind w.r.t. precision of solution
    ASSERT_NEAR(client.y_best, 7.021, 1e-3);
    ASSERT_NEAR(client.x_best[0], 0.757, 1e-3);
}

int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "tester");
  ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}