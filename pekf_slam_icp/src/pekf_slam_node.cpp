// ROS 2 node around PoseEkfSlam: wheel joint states drive the prediction, the IMU yaw corrects the heading, and every
// scan taken after the robot has moved enough becomes a viewpoint that is matched against the overlapping ones.
//
// Subscribes: joint_states (sensor_msgs/JointState), imu (sensor_msgs/Imu), scan (sensor_msgs/LaserScan)
// Publishes:  pekf/pose (PoseWithCovarianceStamped, map frame), pekf/path, pekf/dead_reckoning_path (nav_msgs/Path),
//             pekf/dead_reckoning_pose (PoseStamped), pekf/map_cloud (PointCloud2), pekf/viewpoints (MarkerArray),
//             TF map -> odom -> base
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "pekf_slam_icp/pose_ekf_slam.hpp"
#include "state_estimation/gaussian.hpp"

namespace pekf_slam_icp {
namespace se = state_estimation;

namespace {
/// Yaw of a quaternion (rotation about z, the only one a planar robot has).
double yaw_of(double x, double y, double z, double w) {
    return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

geometry_msgs::msg::Pose to_pose_msg(const Pose2& x) {
    geometry_msgs::msg::Pose p;
    p.position.x = x.x();
    p.position.y = x.y();
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, x.z());
    p.orientation.x = q.x();
    p.orientation.y = q.y();
    p.orientation.z = q.z();
    p.orientation.w = q.w();
    return p;
}

geometry_msgs::msg::TransformStamped to_transform(const Pose2& x, const std::string& parent, const std::string& child,
                                                  const rclcpp::Time& stamp) {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = parent;
    t.child_frame_id = child;
    const auto p = to_pose_msg(x);
    t.transform.translation.x = p.position.x;
    t.transform.translation.y = p.position.y;
    t.transform.rotation = p.orientation;
    return t;
}
}  // namespace

class PekfSlamNode : public rclcpp::Node {
public:
    PekfSlamNode() : rclcpp::Node("pekf_slam") {
        dd_.wheel_radius = declare_parameter("wheel_radius", 0.1);
        dd_.wheel_base = declare_parameter("wheel_separation", 0.5);
        left_joint_ = declare_parameter("left_wheel_joint", std::string("left_wheel_joint"));
        right_joint_ = declare_parameter("right_wheel_joint", std::string("right_wheel_joint"));
        encoder_var_per_m_ = declare_parameter("encoder_variance_per_meter", 2e-4);
        use_imu_ = declare_parameter("use_imu", true);
        const double imu_std = declare_parameter("imu_yaw_stddev", 0.0087);
        imu_var_ = imu_std * imu_std;
        imu_period_ = declare_parameter("imu_update_period", 0.1);
        map_frame_ = declare_parameter("map_frame", std::string("map"));
        odom_frame_ = declare_parameter("odom_frame", std::string("odom"));
        base_frame_ = declare_parameter("base_frame", std::string("base_footprint"));
        publish_tf_ = declare_parameter("publish_tf", true);

        PoseEkfSlamOptions opt;
        opt.viewpoint_distance = declare_parameter("viewpoint_distance", opt.viewpoint_distance);
        opt.viewpoint_angle = declare_parameter("viewpoint_angle", opt.viewpoint_angle);
        opt.overlap_distance = declare_parameter("overlap_distance", opt.overlap_distance);
        opt.max_matches = static_cast<int>(declare_parameter("max_matches", opt.max_matches));
        opt.gate_probability = declare_parameter("gate_probability", opt.gate_probability);
        opt.max_rmse = declare_parameter("icp_max_rmse", opt.max_rmse);
        opt.min_pairs = static_cast<int>(declare_parameter("icp_min_pairs", opt.min_pairs));
        opt.icp.max_correspondence_distance =
            declare_parameter("icp_max_correspondence_distance", opt.icp.max_correspondence_distance);
        const Pose2 x0(declare_parameter("initial_x", 0.0), declare_parameter("initial_y", 0.0),
                       declare_parameter("initial_yaw", 0.0));
        const double s_xy = declare_parameter("initial_position_stddev", 0.01);
        const double s_th = declare_parameter("initial_yaw_stddev", 0.0035);
        slam_.emplace(x0, Eigen::Vector3d(s_xy * s_xy, s_xy * s_xy, s_th * s_th).asDiagonal().toDenseMatrix(), opt);
        odom_ = x0;  // dead reckoning starts at the same pose, so map -> odom starts at identity

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("pekf/pose", 10);
        path_pub_ = create_publisher<nav_msgs::msg::Path>("pekf/path", 10);
        dr_path_pub_ = create_publisher<nav_msgs::msg::Path>("pekf/dead_reckoning_path", 10);
        dr_pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("pekf/dead_reckoning_pose", 10);
        cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("pekf/map_cloud", rclcpp::QoS(1).transient_local());
        marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("pekf/viewpoints", 10);

        joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::JointState::ConstSharedPtr m) { on_joints(*m); });
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "imu", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::Imu::ConstSharedPtr m) { on_imu(*m); });
        scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
            "scan", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::LaserScan::ConstSharedPtr m) { on_scan(*m); });
        timer_ = create_wall_timer(std::chrono::milliseconds(50), [this] { publish_state(); });
        RCLCPP_INFO(get_logger(), "pose-based EKF SLAM ready (IMU heading updates %s)", use_imu_ ? "on" : "off");
    }

private:
    void on_joints(const sensor_msgs::msg::JointState& msg) {
        std::optional<double> left, right;
        for (size_t i = 0; i < msg.name.size() && i < msg.position.size(); ++i) {
            if (msg.name[i] == left_joint_) left = msg.position[i];
            if (msg.name[i] == right_joint_) right = msg.position[i];
        }
        if (!left || !right) return;
        if (!last_left_) {  // first message: only remember the angles
            last_left_ = left;
            last_right_ = right;
            return;
        }
        const double dl = (*left - *last_left_) * dd_.wheel_radius, dr = (*right - *last_right_) * dd_.wheel_radius;
        last_left_ = left;
        last_right_ = right;
        if (dl == 0.0 && dr == 0.0) return;
        const double vl = encoder_var_per_m_ * std::abs(dl), vr = encoder_var_per_m_ * std::abs(dr);
        slam_->predict_wheels(dd_, dl, dr, vl, vr);
        const Eigen::Vector2d arc = dd_.wheel_travel_to_arc(dl, dr);
        odom_ = se::compose(odom_, se::arc_displacement(arc(0), arc(1)));
        last_stamp_ = msg.header.stamp;
    }

    void on_imu(const sensor_msgs::msg::Imu& msg) {
        if (!use_imu_) return;
        const rclcpp::Time t(msg.header.stamp);
        if (last_imu_ && (t - *last_imu_).seconds() < imu_period_) return;
        last_imu_ = t;
        const auto& q = msg.orientation;
        slam_->update_heading(yaw_of(q.x, q.y, q.z, q.w), imu_var_);
    }

    void on_scan(const sensor_msgs::msg::LaserScan& msg) {
        if (!slam_->needs_viewpoint()) return;
        if (!laser_in_base_) {  // static mounting of the lidar, looked up once
            try {
                const auto t = tf_buffer_->lookupTransform(base_frame_, msg.header.frame_id, tf2::TimePointZero);
                const auto& q = t.transform.rotation;
                laser_in_base_ = Pose2(t.transform.translation.x, t.transform.translation.y, yaw_of(q.x, q.y, q.z, q.w));
            } catch (const tf2::TransformException& e) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no transform %s -> %s yet: %s",
                                     base_frame_.c_str(), msg.header.frame_id.c_str(), e.what());
                return;
            }
        }
        MatrixXd scan(msg.ranges.size(), 2);
        Eigen::Index n = 0;
        for (size_t i = 0; i < msg.ranges.size(); ++i) {
            const double r = msg.ranges[i];
            if (!std::isfinite(r) || r < msg.range_min || r > msg.range_max) continue;
            const double a = msg.angle_min + i * msg.angle_increment;
            scan.row(n++) = se::transform_point(*laser_in_base_, Eigen::Vector2d(r * std::cos(a), r * std::sin(a)));
        }
        scan.conservativeResize(n, 2);
        const auto reports = slam_->add_viewpoint(scan);
        int accepted = 0;
        for (const auto& r : reports) accepted += r.accepted ? 1 : 0;
        RCLCPP_DEBUG(get_logger(), "viewpoint %d: %zu candidates, %d accepted", slam_->n_viewpoints() - 1,
                     reports.size(), accepted);
        publish_map(msg.header.stamp);
    }

    void publish_state() {
        if (!last_stamp_) return;
        const rclcpp::Time stamp(*last_stamp_);
        const Pose2 x = slam_->pose();
        const Matrix3d P = slam_->pose_covariance();

        geometry_msgs::msg::PoseWithCovarianceStamped pose;
        pose.header.stamp = stamp;
        pose.header.frame_id = map_frame_;
        pose.pose.pose = to_pose_msg(x);
        const int idx[3] = {0, 1, 5};  // x, y, yaw in the 6x6 row-major covariance
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) pose.pose.covariance[6 * idx[r] + idx[c]] = P(r, c);
        pose_pub_->publish(pose);

        auto append = [&](nav_msgs::msg::Path& path, const Pose2& p) {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = pose.header;
            ps.pose = to_pose_msg(p);
            path.header = pose.header;
            path.poses.push_back(ps);
        };
        append(path_, x);
        append(dr_path_, odom_);
        dr_pose_pub_->publish(dr_path_.poses.back());
        if (++path_counter_ % 10 == 0) {  // 2 Hz is plenty for RViz
            path_pub_->publish(path_);
            dr_path_pub_->publish(dr_path_);
        }

        if (publish_tf_) {
            // map -> odom = x (+) (-)odom, so that map -> odom -> base = the SLAM estimate.
            tf_broadcaster_->sendTransform(to_transform(se::compose(x, se::inverse(odom_)), map_frame_, odom_frame_, stamp));
            tf_broadcaster_->sendTransform(to_transform(odom_, odom_frame_, base_frame_, stamp));
        }
    }

    void publish_map(const builtin_interfaces::msg::Time& stamp) {
        const MatrixXd pts = slam_->map_points();
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header.stamp = stamp;
        cloud.header.frame_id = map_frame_;
        sensor_msgs::PointCloud2Modifier mod(cloud);
        mod.setPointCloud2FieldsByString(1, "xyz");
        mod.resize(pts.rows());
        sensor_msgs::PointCloud2Iterator<float> ix(cloud, "x"), iy(cloud, "y"), iz(cloud, "z");
        for (Eigen::Index i = 0; i < pts.rows(); ++i, ++ix, ++iy, ++iz) {
            *ix = static_cast<float>(pts(i, 0));
            *iy = static_cast<float>(pts(i, 1));
            *iz = 0.1f;
        }
        cloud_pub_->publish(cloud);

        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker dots;
        dots.header = cloud.header;
        dots.ns = "viewpoints";
        dots.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        dots.scale.x = dots.scale.y = dots.scale.z = 0.08;
        dots.color.r = 1.0f;
        dots.color.g = 0.5f;
        dots.color.a = 1.0f;
        for (int i = 0; i < slam_->n_viewpoints(); ++i) {
            geometry_msgs::msg::Point p;
            p.x = slam_->viewpoint(i).x();
            p.y = slam_->viewpoint(i).y();
            dots.points.push_back(p);
            // 95 % position ellipse of the viewpoint, scaled x5 to be visible.
            const se::MatrixXd ring = se::ellipse_points(slam_->viewpoint(i).head<2>(),
                                                         25.0 * slam_->viewpoint_covariance(i).topLeftCorner<2, 2>(),
                                                         0.95, 24);
            visualization_msgs::msg::Marker e;
            e.header = cloud.header;
            e.ns = "ellipses";
            e.id = i;
            e.type = visualization_msgs::msg::Marker::LINE_STRIP;
            e.scale.x = 0.01;
            e.color.b = 1.0f;
            e.color.a = 0.8f;
            for (Eigen::Index k = 0; k <= ring.rows(); ++k) {
                geometry_msgs::msg::Point q;
                q.x = ring(k % ring.rows(), 0);
                q.y = ring(k % ring.rows(), 1);
                e.points.push_back(q);
            }
            markers.markers.push_back(e);
        }
        markers.markers.push_back(dots);
        marker_pub_->publish(markers);
    }

    se::DifferentialDrive dd_;
    std::string left_joint_, right_joint_, map_frame_, odom_frame_, base_frame_;
    double encoder_var_per_m_ = 2e-4, imu_var_ = 1e-4, imu_period_ = 0.1;
    bool use_imu_ = true, publish_tf_ = true;

    std::optional<PoseEkfSlam> slam_;
    Pose2 odom_ = Pose2::Zero();
    std::optional<double> last_left_, last_right_;
    std::optional<rclcpp::Time> last_imu_;
    std::optional<builtin_interfaces::msg::Time> last_stamp_;
    std::optional<Pose2> laser_in_base_;
    nav_msgs::msg::Path path_, dr_path_;
    int path_counter_ = 0;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_, dr_path_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr dr_pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace pekf_slam_icp

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<pekf_slam_icp::PekfSlamNode>());
    rclcpp::shutdown();
    return 0;
}
