#include "actionlib/client/simple_action_client.h"
#include "math.h"
#include "ros/ros.h"
#include "unistd.h"
#include "gtest/gtest.h"

#include "bayesopt4ros/BayesOptAction.h"

using namespace bayesopt4ros;
typedef actionlib::SimpleActionClient<BayesOptAction> Client;


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
    return pow(6.0 * x0 - 2.0, 2) * sin(12.0 * x0 - 4.0);
}


class ExampleClient {
    // ! A demonstration on how to use the BayesOpt server from a C++ node.
    public:
        ExampleClient(std::string server_name) : client_node_(server_name) {
            /*! Constructor of the client that queries the BayesOpt server.

            @param server_name    Name of the server (needs to be consistent with server node).
            */
            ros::NodeHandle nh;
            client_node_.waitForServer();
            
            // First value is just to trigger the server
            BayesOptGoal goal;
            goal.y_new = 0.0;

            // Need boost::bind to pass in the 'this' pointer
            // For details see this tutorial:
            // http://wiki.ros.org/actionlib_tutorials/Tutorials/Writing%20a%20Callback%20Based%20Simple%20Action%20Client
            client_node_.sendGoal(goal, boost::bind(&ExampleClient::bayesOptCallback, this, _1, _2));
        }

        void bayesOptCallback(const actionlib::SimpleClientGoalState& state,
                              const BayesOptResultConstPtr& result) {

            /*! This method is called everytime an iteration of BayesOpt finishes.*/
            x_new_ = result->x_new;
            parametersWereUpdated_ = true;
            std::string result_string = "[Client] x_new = " + vecToString(x_new_, 3);
            ROS_INFO_STREAM(result_string);
        }

        bool checkServer() {
            /*! Small helper that checks if server is online. If not, shutdown. */
            bool isOnline = client_node_.isServerConnected();
            if (isOnline) return true;
            ROS_WARN("[Client] Server seems to be offline. Shutting down.");
            ros::shutdown();
            return false;
        }

        void run() {
            /*! Method that emulates client behavior. */
            if (checkServer() && parametersWereUpdated_)
            {
                // Emulate experiment by querying the objective function
                double y_new = forresterFunction(x_new_);
                parametersWereUpdated_ = false;
                ROS_INFO("[Client] y_new = %.2f", y_new);   
                
                // Keeping track of best point so far for the integration test
                if (y_new < y_best_) {
                    y_best_ = y_new;
                    x_best_ = x_new_;
                }

                // Send new goal to server
                BayesOptGoal goal;
                goal.y_new = y_new;
                client_node_.sendGoal(goal, boost::bind(&ExampleClient::bayesOptCallback, this, _1, _2));
            }
        }

        // Require those for checking the test condition
        double y_best_ = std::numeric_limits<double>::min();
        std::vector<double> x_best_;
    private:
        Client client_node_;
        std::string objective_;
        bool parametersWereUpdated_ = false;
        std::vector<double> x_new_;
};

TEST(ClientTestSuite, testForrester)
{
    // Create client node
    ExampleClient client("BayesOpt");
    ros::Rate loop_rate(1);
    size_t iter = 0;
    while (ros::ok())
    {
        iter++;
        if (iter > 25) break;
        ros::spinOnce();
        client.run();
        loop_rate.sleep();
        ROS_INFO_STREAM(iter);
    }
    ros::shutdown();

    // Be kind w.r.t. precision of solution
    ASSERT_NEAR(client.y_best_, -6.021, 1e-2);
    ASSERT_NEAR(client.x_best_[0], 0.757, 1e-2);
}

int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  ros::init(argc, argv, "tester");
  ros::NodeHandle nh;
  return RUN_ALL_TESTS();
}