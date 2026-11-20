#pragma once
/// \file pose_ekf_slam.hpp
/// Pose-based (delayed-state) EKF SLAM with ICP scan matching. The state holds the current robot pose and the poses
/// of the viewpoints where scans were stored; wheel odometry predicts the current pose, a heading sensor corrects
/// it, and ICP between a new scan and overlapping stored scans gives relative-pose measurements that update the whole
/// state. No ROS in here: the ROS 2 node in pekf_slam_node.cpp only feeds it.
///
/// Theory: learn-state-estimation, chapters 6 (EKF), 8 (gating), 9 (state augmentation) and 10 (scan matching).

#include <Eigen/Dense>
#include <vector>

#include "state_estimation/motion_models.hpp"
#include "state_estimation/pose2.hpp"
#include "state_estimation/scan_matching.hpp"

namespace pekf_slam_icp {

using state_estimation::DifferentialDrive;
using state_estimation::Matrix3d;
using state_estimation::MatrixXd;
using state_estimation::Pose2;
using state_estimation::VectorXd;

struct PoseEkfSlamOptions {
    /// A new viewpoint is stored after the robot has moved this far [m] ...
    double viewpoint_distance = 0.5;
    /// ... or turned this much [rad] since the last one.
    double viewpoint_angle = 0.4;
    /// Viewpoints whose estimated position is within this distance [m], widened by 3 sigma of the current position,
    /// are candidates for scan matching.
    double overlap_distance = 1.5;
    /// At most this many candidates are matched per new viewpoint (the nearest ones).
    int max_matches = 3;
    /// Confidence of the individual-compatibility gate on each ICP measurement (chi2, 3 dof).
    double gate_probability = 0.999;
    /// ICP results with a larger rmse [m] or fewer pairs are discarded before gating.
    double max_rmse = 0.03;
    int min_pairs = 60;
    /// Added to the ICP covariance: the least-squares covariance of a match is optimistic.
    Matrix3d icp_covariance_floor = Eigen::Vector3d(1e-4, 1e-4, 3e-5).asDiagonal();
    state_estimation::IcpOptions icp;
};

/// Outcome of matching the new viewpoint against one stored viewpoint.
struct MatchReport {
    int viewpoint = -1;       ///< index of the stored viewpoint
    bool converged = false;   ///< ICP converged with a small rmse and enough pairs
    bool accepted = false;    ///< passed the gate and updated the filter
    double mahalanobis2 = 0;  ///< squared Mahalanobis distance of the innovation
    double rmse = 0;
    Pose2 measurement = Pose2::Zero();  ///< ICP: pose of the new scan's frame in the stored scan's frame
};

class PoseEkfSlam {
public:
    PoseEkfSlam(const Pose2& x0, const Matrix3d& P0, const PoseEkfSlamOptions& options = {});

    /// Prediction with a robot-frame displacement u ~ N(u, Q): only the current-pose block and its cross-covariances
    /// change (the viewpoints do not move).
    void predict_displacement(const Pose2& u, const Matrix3d& Q);
    /// Prediction from the wheel travel since the last call (arc model, encoder variances).
    void predict_wheels(const DifferentialDrive& dd, double d_left, double d_right, double var_left, double var_right);

    /// Heading measurement (IMU yaw or compass) with variance var: z = theta + v.
    void update_heading(double yaw, double var);

    /// True when the robot has moved or turned enough since the last stored viewpoint (or none is stored yet).
    bool needs_viewpoint() const;

    /// Store the current pose as a new viewpoint with its scan (robot frame, m x 2), then match the scan against the
    /// overlapping stored viewpoints and update the filter with every match that passes the gate.
    std::vector<MatchReport> add_viewpoint(const MatrixXd& scan);

    Pose2 pose() const { return y_.head<3>(); }
    Matrix3d pose_covariance() const { return P_.topLeftCorner<3, 3>(); }
    int n_viewpoints() const { return static_cast<int>(scans_.size()); }
    Pose2 viewpoint(int i) const { return y_.segment<3>(3 + 3 * i); }
    Matrix3d viewpoint_covariance(int i) const { return P_.block<3, 3>(3 + 3 * i, 3 + 3 * i); }
    const std::vector<MatrixXd>& scans() const { return scans_; }
    const VectorXd& state() const { return y_; }
    const MatrixXd& covariance() const { return P_; }
    /// Every stored scan in the map frame at its current viewpoint estimate, stacked.
    MatrixXd map_points() const;
    int n_accepted_matches() const { return n_accepted_; }

private:
    /// Clone the current pose into the state: y = [x; v_1..v_n; x], with the exact cross-covariances.
    void clone_current_pose();
    /// EKF update with a measurement z of the pose of the current robot frame seen from viewpoint i.
    bool update_relative(int i, const Pose2& z, const Matrix3d& R, double& d2);

    PoseEkfSlamOptions options_;
    VectorXd y_;
    MatrixXd P_;
    std::vector<MatrixXd> scans_;
    int n_accepted_ = 0;
};

}  // namespace pekf_slam_icp
