// LapFollower on an ideal unicycle in the office of pekf_slam_icp_gazebo/worlds/office.sdf: it must finish its laps
// on the default route without coming close to any wall, and back off when the robot is held in place.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "pekf_slam_icp/lap_follower.hpp"

using pekf_slam_icp::LapFollower;
using pekf_slam_icp::LapFollowerParams;
using pekf_slam_icp::Point2;

namespace {

// Wall centre lines (ax, ay, bx, by), as in scripts/make_office_world.py; the walls are 0.1 m thick.
const std::vector<std::array<double, 4>> kWalls = {
    {-8, -6, 8, -6}, {8, -6, 8, 6}, {8, 6, -8, 6}, {-8, 6, -8, -6},
    {-3, -6, -3, -4.9}, {-3, -3.1, -3, 3.7}, {-3, 5.3, -3, 6},
    {3, -6, 3, -4.9}, {3, -3.3, 3, 3.4}, {3, 5.0, 3, 6},
    {-1, -2.5, 1, -2.5}, {0, 2.5, 1, 3.3}, {-4.6, 0.5, -3.8, 1.3}, {3.8, -2, 4.6, -1.2},
    {-7.5, -1, -7.5, 1}, {7.4, 2, 7.4, 3.5}};

double distance_to_walls(const Point2& p) {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& [ax, ay, bx, by] : kWalls) {
        const double dx = bx - ax, dy = by - ay;
        const double u = std::clamp(((p.x - ax) * dx + (p.y - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0);
        best = std::min(best, std::hypot(p.x - ax - u * dx, p.y - ay - u * dy) - 0.05);
    }
    return best;
}

const std::vector<Point2> kRoute = {{6.3, -4.05}, {6.3, 4.35}, {-6.3, 4.35}, {-6.3, -4.05}};

}  // namespace

TEST(LapFollower, FinishesTwoOfficeLapsClearOfTheWalls) {
    LapFollowerParams p;  // node defaults
    LapFollower f(kRoute, p);
    Point2 pos{-5.5, -4.0};  // spawn pose of sim.launch.py
    double yaw = 0.0, travelled = 0.0, min_clearance = std::numeric_limits<double>::infinity();
    const double dt = 0.02;
    double t = 0.0;
    for (; t < 600.0 && !f.finished(); t += dt) {
        const auto c = f.step(pos, yaw, t);
        pos.x += c.v * std::cos(yaw) * dt;
        pos.y += c.v * std::sin(yaw) * dt;
        yaw += c.omega * dt;
        travelled += std::abs(c.v) * dt;
        min_clearance = std::min(min_clearance, distance_to_walls(pos));
    }
    EXPECT_TRUE(f.finished());
    EXPECT_EQ(f.laps_done(), 2);
    EXPECT_EQ(f.backoffs(), 0);
    // Two laps of the 12.6 m x 8.4 m loop, less the 0.8 m between spawn and the loop's start corner and the
    // look-ahead (0.8 m) by which it stops early; corner cutting by the look-ahead shortens it a little more.
    EXPECT_NEAR(travelled, 2 * 2 * (12.6 + 8.4) - 0.8 - p.lookahead, 4.0);
    // The robot's corners are at most 0.42 m from its centre; keep at least 0.15 m to spare.
    EXPECT_GT(min_clearance, 0.42 + 0.15) << "closest approach to a wall surface";
    RecordProperty("min_clearance_m", std::to_string(min_clearance));
}

TEST(LapFollower, BacksOffWhenHeldInPlace) {
    LapFollowerParams p;
    LapFollower f(kRoute, p);
    const Point2 pos{-5.5, -4.0};
    double t = 0.0;
    auto c = f.step(pos, 0.0, t);
    EXPECT_GT(c.v, 0.0);
    for (; t < p.stuck_time + 0.5 && f.backoffs() == 0; t += 0.05) c = f.step(pos, 0.0, t);
    ASSERT_EQ(f.backoffs(), 1);
    EXPECT_LT(c.v, 0.0);
    EXPECT_GT(t, p.stuck_time);
    // Reverses for backoff_time, then drives forward again.
    c = f.step(pos, 0.0, t + 0.5 * p.backoff_time);
    EXPECT_LT(c.v, 0.0);
    c = f.step(pos, 0.0, t + p.backoff_time + 0.1);
    EXPECT_GT(c.v, 0.0);
}

TEST(LapFollower, SlowsDownInFrontOfAnObstacle) {
    LapFollowerParams p;
    LapFollower free_run(kRoute, p), blocked(kRoute, p);
    const auto c_free = free_run.step({-5.5, -4.0}, 0.0, 0.0);
    const auto c_near = blocked.step({-5.5, -4.0}, 0.0, 0.0, /*front_clearance=*/0.2);
    EXPECT_NEAR(c_near.v, p.min_speed_scale * c_free.v, 1e-12);
    EXPECT_DOUBLE_EQ(c_near.omega, c_free.omega);
}
