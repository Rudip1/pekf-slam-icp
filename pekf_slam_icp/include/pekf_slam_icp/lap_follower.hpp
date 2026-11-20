// ROS-free core of the scripted lap driver: follows a closed route of straight segments with a carrot (look-ahead)
// point, slows down when the scan shows an obstacle ahead, and backs off when it stops making progress.
//
// Following the segments, not just aiming at the next waypoint, keeps the robot on the route through the doorways:
// aiming at a point cuts every corner and can clip a door jamb, which is how the first ROS 2 runs got stuck.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace pekf_slam_icp {

struct Point2 {
    double x = 0.0, y = 0.0;
};

struct LapFollowerParams {
    int laps = 2;
    double speed = 0.4;             // cruise speed [m/s]
    double lookahead = 0.8;         // carrot distance along the route [m]
    double heading_gain = 1.5;      // omega = gain * heading error [1/s]
    double max_angular_speed = 0.8; // [rad/s]
    double slow_distance = 0.8;     // start slowing when the nearest return ahead is closer than this [m]
    double stop_distance = 0.3;     // ... and creep at min_speed_scale below this [m]
    double min_speed_scale = 0.25;
    double stuck_time = 4.0;        // no progress of stuck_progress within this time = stuck [s]
    double stuck_progress = 0.10;   // [m]
    double backoff_time = 1.5;      // reverse for this long when stuck [s]
    double backoff_speed = 0.15;    // [m/s]
};

struct Command {
    double v = 0.0, omega = 0.0;
};

class LapFollower {
public:
    LapFollower(std::vector<Point2> route, LapFollowerParams p) : route_(std::move(route)), p_(p) {}

    // One control step. `front_clearance` is the nearest scan return in the forward cone (infinity if none).
    Command step(const Point2& pos, double yaw, double t, double front_clearance = INFINITY) {
        if (finished_ || route_.empty()) return {};
        if (!started_) {
            start_ = pos;  // the first segment runs from where the robot is to the first waypoint
            started_ = true;
            last_progress_t_ = t;
        }
        if (t < backoff_until_) return {-p_.backoff_speed, 0.0};

        // Advance to the next segment once the projection is within one look-ahead of the segment's end.
        Point2 a = from(), b = route_[seg_];
        double len = dist(a, b), s = project(pos, a, b);
        while (s >= len - p_.lookahead) {
            if (seg_ + 1 == route_.size()) {   // reached the last waypoint: one lap done
                if (++lap_ >= p_.laps) {
                    finished_ = true;
                    return {};
                }
            }
            seg_ = (seg_ + 1) % route_.size();
            first_segment_ = false;
            done_before_ += len;
            a = from();
            b = route_[seg_];
            len = dist(a, b);
            s = project(pos, a, b);
        }

        // Stuck detection on the distance travelled along the route.
        const double progress = done_before_ + std::max(0.0, s);
        if (progress > best_progress_ + p_.stuck_progress) {
            best_progress_ = progress;
            last_progress_t_ = t;
        } else if (t - last_progress_t_ > p_.stuck_time) {
            ++backoffs_;
            backoff_until_ = t + p_.backoff_time;
            last_progress_t_ = backoff_until_;  // give it stuck_time after the back-off to get going again
            return {-p_.backoff_speed, 0.0};
        }

        const double c = std::min(std::max(0.0, s) + p_.lookahead, len) / len;
        const Point2 carrot{a.x + c * (b.x - a.x), a.y + c * (b.y - a.y)};
        const double err = wrap(std::atan2(carrot.y - pos.y, carrot.x - pos.x) - yaw);
        const double clear = std::clamp((front_clearance - p_.stop_distance) / (p_.slow_distance - p_.stop_distance),
                                        p_.min_speed_scale, 1.0);
        Command cmd;
        cmd.omega = std::clamp(p_.heading_gain * err, -p_.max_angular_speed, p_.max_angular_speed);
        cmd.v = p_.speed * clear * std::max(0.0, std::cos(err));  // slow down while turning towards the carrot
        return cmd;
    }

    bool finished() const { return finished_; }
    int laps_done() const { return lap_; }
    int backoffs() const { return backoffs_; }

    static double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

private:
    Point2 from() const { return first_segment_ ? start_ : route_[(seg_ + route_.size() - 1) % route_.size()]; }
    static double dist(const Point2& a, const Point2& b) { return std::hypot(b.x - a.x, b.y - a.y); }
    static double project(const Point2& p, const Point2& a, const Point2& b) {
        const double len = dist(a, b);
        return len < 1e-9 ? 0.0 : ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / len;
    }

    std::vector<Point2> route_;
    LapFollowerParams p_;
    Point2 start_;
    std::size_t seg_ = 0;
    bool started_ = false, first_segment_ = true, finished_ = false;
    int lap_ = 0, backoffs_ = 0;
    double done_before_ = 0.0, best_progress_ = 0.0, last_progress_t_ = 0.0, backoff_until_ = -INFINITY;
};

}  // namespace pekf_slam_icp
