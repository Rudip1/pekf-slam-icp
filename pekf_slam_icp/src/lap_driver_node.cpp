// Scripted driver for reproducible runs: follows a closed route through the office for a number of laps
// (pekf_slam_icp::LapFollower) and publishes `lap_driver/finished` when done. It steers on the simulator's ground truth
// so that every run follows the same route; the estimator never sees that topic. The scan is used only to slow down
// when something is close ahead.
//
// Subscribes: ground_truth/odometry (nav_msgs/Odometry), scan (sensor_msgs/LaserScan)
// Publishes:  cmd_vel (geometry_msgs/Twist), lap_driver/finished (std_msgs/Empty, transient local)
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/empty.hpp>

#include "pekf_slam_icp/lap_follower.hpp"

using pekf_slam_icp::LapFollower;
using pekf_slam_icp::LapFollowerParams;
using pekf_slam_icp::Point2;

class LapDriver : public rclcpp::Node {
public:
    LapDriver() : rclcpp::Node("lap_driver") {
        // Default route: the office loop through the four doorways (pekf_slam_icp_gazebo/worlds/office.sdf). The two
        // corridors run at the mean height of their door centres (lower doors y = -4.0 / -4.1, upper 4.5 / 4.2).
        const std::vector<double> flat = declare_parameter(
            "waypoints", std::vector<double>{6.3, -4.05, 6.3, 4.35, -6.3, 4.35, -6.3, -4.05});
        std::vector<Point2> route;
        for (size_t i = 0; i + 1 < flat.size(); i += 2) route.push_back({flat[i], flat[i + 1]});
        LapFollowerParams p;
        p.laps = static_cast<int>(declare_parameter("laps", p.laps));
        p.speed = declare_parameter("speed", p.speed);
        p.lookahead = declare_parameter("lookahead", p.lookahead);
        p.heading_gain = declare_parameter("heading_gain", p.heading_gain);
        p.max_angular_speed = declare_parameter("max_angular_speed", p.max_angular_speed);
        p.slow_distance = declare_parameter("slow_distance", p.slow_distance);
        p.stop_distance = declare_parameter("stop_distance", p.stop_distance);
        p.stuck_time = declare_parameter("stuck_time", p.stuck_time);
        p.backoff_time = declare_parameter("backoff_time", p.backoff_time);
        cone_ = declare_parameter("front_cone_half_angle", 0.5);
        follower_ = std::make_unique<LapFollower>(route, p);

        cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
        done_pub_ = create_publisher<std_msgs::msg::Empty>("lap_driver/finished", rclcpp::QoS(1).transient_local());
        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::LaserScan::ConstSharedPtr m) { on_scan(*m); });
        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "ground_truth/odometry", 10, [this](nav_msgs::msg::Odometry::ConstSharedPtr m) { step(*m); });
        RCLCPP_INFO(get_logger(), "driving %d laps over %zu waypoints at %.2f m/s", p.laps, route.size(), p.speed);
    }

private:
    void on_scan(const sensor_msgs::msg::LaserScan& scan) {
        double nearest = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < scan.ranges.size(); ++i) {
            const double angle = scan.angle_min + static_cast<double>(i) * scan.angle_increment;
            const double r = scan.ranges[i];
            if (std::abs(angle) <= cone_ && std::isfinite(r) && r >= scan.range_min && r <= scan.range_max)
                nearest = std::min(nearest, r);
        }
        front_clearance_ = nearest;
    }

    void step(const nav_msgs::msg::Odometry& odom) {
        if (follower_->finished()) return;
        const auto& q = odom.pose.pose.orientation;
        const double yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        const int backoffs = follower_->backoffs();
        const auto c = follower_->step({odom.pose.pose.position.x, odom.pose.pose.position.y}, yaw,
                                       rclcpp::Time(odom.header.stamp).seconds(), front_clearance_);
        if (follower_->backoffs() > backoffs)
            RCLCPP_WARN(get_logger(), "no progress at (%.2f, %.2f): backing off", odom.pose.pose.position.x,
                        odom.pose.pose.position.y);
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = c.v;
        cmd.angular.z = c.omega;
        cmd_pub_->publish(cmd);
        if (follower_->finished()) {
            done_pub_->publish(std_msgs::msg::Empty{});
            RCLCPP_INFO(get_logger(), "finished %d laps (%d back-offs)", follower_->laps_done(), follower_->backoffs());
        }
    }

    std::unique_ptr<LapFollower> follower_;
    double cone_ = 0.5, front_clearance_ = std::numeric_limits<double>::infinity();
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr done_pub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LapDriver>());
    rclcpp::shutdown();
    return 0;
}
