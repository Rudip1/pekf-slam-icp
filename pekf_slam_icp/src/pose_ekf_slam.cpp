#include "pekf_slam_icp/pose_ekf_slam.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "state_estimation/gaussian.hpp"

namespace pekf_slam_icp {

namespace se = state_estimation;

namespace {
/// Wrap the heading of the current pose and of every viewpoint.
void wrap_headings(VectorXd& y) {
    for (Eigen::Index k = 2; k < y.size(); k += 3) y(k) = se::wrap_angle(y(k));
}

/// Joseph-form covariance update, symmetric by construction.
void joseph_update(MatrixXd& P, const MatrixXd& K, const MatrixXd& H, const MatrixXd& R) {
    const MatrixXd I_KH = MatrixXd::Identity(P.rows(), P.cols()) - K * H;
    P = I_KH * P * I_KH.transpose() + K * R * K.transpose();
    P = 0.5 * (P + P.transpose());
}
}  // namespace

PoseEkfSlam::PoseEkfSlam(const Pose2& x0, const Matrix3d& P0, const PoseEkfSlamOptions& options)
    : options_(options), y_(x0), P_(P0) {}

void PoseEkfSlam::predict_displacement(const Pose2& u, const Matrix3d& Q) {
    // Only the current pose moves: P_xx = F P_xx F^T + W Q W^T, P_xv = F P_xv (learn-state-estimation eq. 9.4).
    const Pose2 x = pose();
    const Matrix3d F = se::compose_jacobian_1(x, u), W = se::compose_jacobian_2(x, u);
    y_.head<3>() = se::compose(x, u);
    y_(2) = se::wrap_angle(y_(2));
    const Eigen::Index n = P_.cols();
    P_.topLeftCorner<3, 3>() = F * P_.topLeftCorner<3, 3>() * F.transpose() + W * Q * W.transpose();
    if (n > 3) {
        P_.topRightCorner(3, n - 3) = F * P_.topRightCorner(3, n - 3);
        P_.bottomLeftCorner(n - 3, 3) = P_.topRightCorner(3, n - 3).transpose();
    }
}

void PoseEkfSlam::predict_wheels(const DifferentialDrive& dd, double d_left, double d_right, double var_left,
                                 double var_right) {
    const Eigen::Vector2d arc = dd.wheel_travel_to_arc(d_left, d_right);
    predict_displacement(se::arc_displacement(arc(0), arc(1)),
                         se::wheel_displacement_covariance(dd, d_left, d_right, var_left, var_right));
}

void PoseEkfSlam::update_heading(double yaw, double var) {
    MatrixXd H = MatrixXd::Zero(1, y_.size());
    H(0, 2) = 1.0;
    const double nu = se::wrap_angle(yaw - y_(2));
    const double S = P_(2, 2) + var;
    const MatrixXd K = P_.col(2) / S;
    y_ += K * nu;
    wrap_headings(y_);
    joseph_update(P_, K, H, MatrixXd::Constant(1, 1, var));
}

bool PoseEkfSlam::needs_viewpoint() const {
    if (scans_.empty()) return true;
    const Pose2 last = viewpoint(n_viewpoints() - 1), x = pose();
    return (x.head<2>() - last.head<2>()).norm() > options_.viewpoint_distance ||
           std::abs(se::wrap_angle(x.z() - last.z())) > options_.viewpoint_angle;
}

void PoseEkfSlam::clone_current_pose() {
    // y+ = [y; x], P+ = [P, P_{:,x}; P_{x,:}, P_xx]: the clone is exactly the current pose, fully correlated with it.
    const Eigen::Index n = y_.size();
    VectorXd y(n + 3);
    y << y_, y_.head<3>();
    MatrixXd P(n + 3, n + 3);
    P.topLeftCorner(n, n) = P_;
    P.topRightCorner(n, 3) = P_.leftCols<3>();
    P.bottomLeftCorner(3, n) = P_.topRows<3>();
    P.bottomRightCorner<3, 3>() = P_.topLeftCorner<3, 3>();
    y_ = std::move(y);
    P_ = std::move(P);
}

bool PoseEkfSlam::update_relative(int i, const Pose2& z, const Matrix3d& R, double& d2) {
    // h(y) = (-)v_i (+) x; dh/dv_i = J1(+)((-)v_i, x) J(-)(v_i), dh/dx = J2(+)((-)v_i, x).
    const Pose2 v = viewpoint(i), x = pose();
    const Pose2 inv_v = se::inverse(v);
    const Pose2 h = se::compose(inv_v, x);
    MatrixXd H = MatrixXd::Zero(3, y_.size());
    H.leftCols<3>() = se::compose_jacobian_2(inv_v, x);
    H.block<3, 3>(0, 3 + 3 * i) = se::compose_jacobian_1(inv_v, x) * se::inverse_jacobian(v);
    Eigen::Vector3d nu = z - h;
    nu.z() = se::wrap_angle(nu.z());
    const Matrix3d S = H * P_ * H.transpose() + R;
    const Eigen::LDLT<Matrix3d> S_ldlt(S);
    d2 = nu.dot(S_ldlt.solve(nu));
    if (!(d2 < se::chi2_quantile(options_.gate_probability, 3))) return false;  // individual compatibility
    const MatrixXd K = S_ldlt.solve(H * P_).transpose();                     // P H^T S^-1 (S, P symmetric)
    y_ += K * nu;
    wrap_headings(y_);
    joseph_update(P_, K, H, R);
    return true;
}

std::vector<MatchReport> PoseEkfSlam::add_viewpoint(const MatrixXd& scan) {
    clone_current_pose();
    scans_.push_back(scan);
    std::vector<MatchReport> reports;
    const int n = n_viewpoints();
    if (n < 2 || scan.rows() < 3) return reports;

    // Candidates: stored viewpoints near the current estimate, the radius widened by its uncertainty.
    const Pose2 x = pose();
    const double sigma =
        std::sqrt(Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(P_.topLeftCorner<2, 2>()).eigenvalues()(1));
    std::vector<std::pair<double, int>> candidates;
    for (int i = 0; i + 1 < n; ++i) {
        const double d = (viewpoint(i).head<2>() - x.head<2>()).norm();
        if (d < options_.overlap_distance + 3.0 * sigma && scans_[i].rows() >= 3) candidates.emplace_back(d, i);
    }
    std::sort(candidates.begin(), candidates.end());
    if (static_cast<int>(candidates.size()) > options_.max_matches) candidates.resize(options_.max_matches);

    for (const auto& [dist, i] : candidates) {
        MatchReport report;
        report.viewpoint = i;
        const Pose2 predicted = se::between(viewpoint(i), pose());
        const se::IcpResult r = se::icp_point_to_line(scans_[i], scan, predicted, options_.icp);
        report.rmse = r.rmse;
        report.measurement = r.pose;
        report.converged = r.converged && r.rmse < options_.max_rmse && r.n_pairs >= options_.min_pairs;
        if (report.converged)
            report.accepted =
                update_relative(i, r.pose, r.covariance + options_.icp_covariance_floor, report.mahalanobis2);
        n_accepted_ += report.accepted ? 1 : 0;
        reports.push_back(report);
    }
    return reports;
}

MatrixXd PoseEkfSlam::map_points() const {
    Eigen::Index total = 0;
    for (const auto& s : scans_) total += s.rows();
    MatrixXd out(total, 2);
    Eigen::Index row = 0;
    for (int i = 0; i < n_viewpoints(); ++i) {
        out.middleRows(row, scans_[i].rows()) = se::transform_points(viewpoint(i), scans_[i]);
        row += scans_[i].rows();
    }
    return out;
}

}  // namespace pekf_slam_icp
