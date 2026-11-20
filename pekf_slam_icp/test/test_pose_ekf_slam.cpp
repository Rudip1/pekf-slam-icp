// Tests of the pose-based EKF, ROS-free: prediction equals dead reckoning; cloning is exact; the heading update is the
// scalar Kalman update; a wrong relative measurement is gated out; and a run in the simulated office (learn-state-
// estimation's lidar simulator) stays accurate and consistent against ground truth.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "pekf_slam_icp/pose_ekf_slam.hpp"
#include "state_estimation/gaussian.hpp"
#include "state_estimation/simulation.hpp"

using namespace pekf_slam_icp;
namespace se = state_estimation;

namespace {

struct OfficeResult {
    double ate = 0.0;            ///< RMS position error of the current pose over the run [m]
    double dead_reckoning_ate = 0.0;
    double mean_nees = 0.0;      ///< mean NEES of the current pose (3 = consistent)
    int viewpoints = 0;
    int accepted = 0;
    int wrong_accepted = 0;      ///< accepted ICP measurements more than 5 cm from the truth
};

/// Two laps of the office; wheel encoders every step, IMU yaw (sigma 0.5 deg) every step, a scan every 10 steps.
OfficeResult run_office(int seed, bool use_imu) {
    const se::DifferentialDrive dd{0.1, 0.5, 0};
    const double k = 2e-4, dt = 0.1, sigma_yaw = 0.5 * 3.14159265358979 / 180.0;
    const Pose2 x0(-5.5, -4.0, 0.0);
    se::Rng rng(seed);
    const se::LidarRun run = se::simulate_lidar_run(x0, se::office_lap_controls(2, dt), dt, dd, k,
                                                    se::office_world(), 10, 180, 8.0, 0.01, rng);
    const Matrix3d P0 = Eigen::Vector3d(1e-4, 1e-4, 1e-5).asDiagonal();  // the start pose, known to 1 cm / 0.2 deg
    PoseEkfSlam slam(x0, P0);
    const se::DeadReckoningRun dr = se::run_dead_reckoning(x0, P0, run.wheel_travel, dd, k);
    OfficeResult out;
    double se2 = 0.0, dr2 = 0.0, nees = 0.0;
    std::vector<Pose2> viewpoint_truth{x0};
    slam.add_viewpoint(run.scans[0]);
    const int steps = static_cast<int>(run.wheel_travel.rows());
    for (int s = 0; s < steps; ++s) {
        const double dl = run.wheel_travel(s, 0), dr_ = run.wheel_travel(s, 1);
        slam.predict_wheels(dd, dl, dr_, k * std::abs(dl), k * std::abs(dr_));
        const Pose2 truth = run.truth.row(s + 1).transpose();
        if (use_imu) slam.update_heading(truth.z() + rng.normal(0.0, sigma_yaw), sigma_yaw * sigma_yaw);
        if ((s + 1) % 10 == 0 && slam.needs_viewpoint()) {
            viewpoint_truth.push_back(truth);
            for (const MatchReport& m : slam.add_viewpoint(run.scans[(s + 1) / 10])) {
                if (!m.accepted) continue;
                ++out.accepted;
                const Pose2 z_true = se::between(viewpoint_truth[m.viewpoint], truth);
                if ((m.measurement.head<2>() - z_true.head<2>()).norm() > 0.05) ++out.wrong_accepted;
            }
        }
        Eigen::Vector3d e = slam.pose() - truth;
        e.z() = se::wrap_angle(e.z());
        se2 += e.head<2>().squaredNorm();
        nees += e.dot(slam.pose_covariance().ldlt().solve(e));  // NEES of the current pose
        dr2 += (dr.poses.row(s + 1).head<2>() - truth.head<2>().transpose()).squaredNorm();
    }
    out.ate = std::sqrt(se2 / steps);
    out.dead_reckoning_ate = std::sqrt(dr2 / steps);
    out.mean_nees = nees / steps;
    out.viewpoints = slam.n_viewpoints();
    return out;
}

}  // namespace

TEST(PoseEkfSlam, PredictionEqualsDeadReckoning) {
    const se::DifferentialDrive dd{0.1, 0.5, 0};
    const double k = 2e-4;
    se::Rng rng(1);
    const MatrixXd u = se::circle_controls(0.4, 2.0, 200);
    const MatrixXd wheels = se::simulate_wheel_travel(u, 0.1, dd, k, rng);
    const Pose2 x0(1.0, 2.0, 0.3);
    const Matrix3d P0 = Eigen::Vector3d(0.01, 0.02, 0.003).asDiagonal();
    PoseEkfSlam slam(x0, P0);
    for (int s = 0; s < wheels.rows(); ++s)
        slam.predict_wheels(dd, wheels(s, 0), wheels(s, 1), k * std::abs(wheels(s, 0)), k * std::abs(wheels(s, 1)));
    const se::DeadReckoningRun dr = se::run_dead_reckoning(x0, P0, wheels, dd, k);
    EXPECT_LT((slam.pose() - dr.poses.bottomRows<1>().transpose()).norm(), 1e-12);
    EXPECT_LT((slam.pose_covariance() - dr.covariances.back()).norm(), 1e-12);
}

TEST(PoseEkfSlam, CloningIsExactAndPredictionLeavesViewpointsAlone) {
    PoseEkfSlam slam(Pose2(0.5, -0.2, 0.4), Eigen::Vector3d(0.02, 0.03, 0.01).asDiagonal());
    slam.add_viewpoint(MatrixXd::Zero(0, 2));
    const MatrixXd& P = slam.covariance();
    ASSERT_EQ(P.rows(), 6);
    EXPECT_LT((slam.viewpoint(0) - slam.pose()).norm(), 1e-15);
    EXPECT_LT((P.topRightCorner<3, 3>() - P.topLeftCorner<3, 3>()).norm(), 1e-15);
    EXPECT_LT((P.bottomRightCorner<3, 3>() - P.topLeftCorner<3, 3>()).norm(), 1e-15);
    const Pose2 v = slam.viewpoint(0);
    const Matrix3d Pv = slam.viewpoint_covariance(0);
    slam.predict_displacement(Pose2(0.3, 0.0, 0.1), Eigen::Vector3d(1e-3, 1e-3, 1e-4).asDiagonal());
    EXPECT_EQ(slam.viewpoint(0), v);
    EXPECT_EQ(slam.viewpoint_covariance(0), Pv);
}

TEST(PoseEkfSlam, HeadingUpdateIsTheScalarKalmanUpdate) {
    const double var_theta = 0.04, var_z = 0.01;
    PoseEkfSlam slam(Pose2(0.0, 0.0, 0.2), Eigen::Vector3d(0.01, 0.01, var_theta).asDiagonal());
    slam.update_heading(0.5, var_z);
    const double gain = var_theta / (var_theta + var_z);
    EXPECT_NEAR(slam.pose().z(), 0.2 + gain * 0.3, 1e-12);
    EXPECT_NEAR(slam.pose_covariance()(2, 2), (1.0 - gain) * var_theta, 1e-12);
    EXPECT_NEAR(slam.pose().x(), 0.0, 1e-15);  // x, y uncorrelated with the heading here: unchanged
}

TEST(PoseEkfSlam, WrongMatchIsGatedOut) {
    // Two scans of the same place from slightly different poses; the second scan is matched against the first. A
    // corrupted copy of the second scan (shifted by 0.6 m) produces a match that the filter must reject.
    const auto walls = se::office_world();
    se::Rng rng(4);
    const Pose2 xa(-5.5, -4.0, 0.0), u(0.4, 0.0, 0.05);
    const Pose2 xb = se::compose(xa, u);
    const MatrixXd scan_a = se::simulate_scan(xa, walls, 360, 6.28, 8.0, 0.01, rng);
    const MatrixXd scan_b = se::simulate_scan(xb, walls, 360, 6.28, 8.0, 0.01, rng);
    const Matrix3d Q = Eigen::Vector3d(1e-4, 1e-4, 1e-5).asDiagonal();

    PoseEkfSlamOptions opt;
    opt.viewpoint_distance = 0.1;
    PoseEkfSlam good(xa, Matrix3d::Zero(), opt), bad(xa, Matrix3d::Zero(), opt);
    good.add_viewpoint(scan_a);
    bad.add_viewpoint(scan_a);
    good.predict_displacement(u, Q);
    bad.predict_displacement(u, Q);
    const std::vector<MatchReport> rg = good.add_viewpoint(scan_b);
    const std::vector<MatchReport> rb = bad.add_viewpoint(se::transform_points(Pose2(0.6, 0.0, 0.0), scan_b));
    ASSERT_EQ(rg.size(), 1u);
    EXPECT_TRUE(rg[0].accepted);
    EXPECT_LT((good.pose().head<2>() - xb.head<2>()).norm(), 0.02);
    ASSERT_EQ(rb.size(), 1u);
    EXPECT_FALSE(rb[0].accepted);
    EXPECT_GT(rb[0].mahalanobis2, se::chi2_quantile(0.999, 3));
}

TEST(PoseEkfSlam, TwoLapsOfTheOfficeStayAccurateAndConsistent) {
    // With IMU heading updates the heading stays observable, the linearisation stays good and the map is accurate.
    // Without them the heading uncertainty grows large before the first loop closes and the EKF, which cannot
    // re-linearise, keeps part of the error (learn-state-estimation 9.7): only a clear gain over dead reckoning and
    // no wrong match are required there.
    for (bool imu : {true, false}) {
        for (int seed : {3, 5, 7}) {
            const OfficeResult r = run_office(seed, imu);
            SCOPED_TRACE(::testing::Message() << "imu " << imu << " seed " << seed << ": ATE " << r.ate
                                              << " m (dead reckoning " << r.dead_reckoning_ate << " m), NEES "
                                              << r.mean_nees << ", " << r.viewpoints << " viewpoints, "
                                              << r.accepted << " matches");
            std::printf("imu %d seed %d: ATE %.4f m (DR %.3f m), mean NEES %.2f, %d viewpoints, %d matches, %d wrong\n",
                        imu, seed, r.ate, r.dead_reckoning_ate, r.mean_nees, r.viewpoints, r.accepted,
                        r.wrong_accepted);
            EXPECT_EQ(r.wrong_accepted, 0);
            EXPECT_GT(r.accepted, 100);
            if (imu) {
                EXPECT_LT(r.ate, 0.1);
                EXPECT_LT(r.mean_nees, 9.0);
            } else {
                EXPECT_LT(r.ate, 0.7 * r.dead_reckoning_ate);
            }
        }
    }
}
