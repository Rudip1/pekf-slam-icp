#!/usr/bin/env python3
"""Evaluate a simulated run against ground truth.

Records the SLAM pose (pekf/pose, with covariance), the dead-reckoning pose (pekf/dead_reckoning_pose) and the
simulator's ground truth (ground_truth/odometry). Each estimate is compared with the ground truth interpolated at the
estimate's own time stamp, so that the 50 Hz sampling of the ground truth does not show up as error. When the scripted driver publishes lap_driver/finished, it writes
a CSV, prints the absolute trajectory error (RMS position error), the final errors and the mean NEES, and exits.

Parameters: output (CSV path, default pekf_run.csv), settle_time (s to wait after the driver stops, default 2.0).
"""
import bisect
import csv
import math

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Empty


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Evaluator(Node):
    def __init__(self):
        super().__init__("pekf_evaluator")
        self.output = self.declare_parameter("output", "pekf_run.csv").value
        self.settle = self.declare_parameter("settle_time", 2.0).value
        self.truth_t, self.truth = [], []  # histories: stamps and (x, y, yaw)
        self.dr_t, self.dr = [], []
        self.rows, self.pending = [], []
        self.done = False
        self.create_subscription(Odometry, "ground_truth/odometry", self.on_truth, 50)
        self.create_subscription(PoseStamped, "pekf/dead_reckoning_pose", self.on_dr, 50)
        self.create_subscription(PoseWithCovarianceStamped, "pekf/pose", self.on_slam, 50)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Empty, "lap_driver/finished", self.on_finished, latched)

    def on_truth(self, m):
        p = m.pose.pose
        self.truth_t.append(stamp_of(m.header))
        self.truth.append((p.position.x, p.position.y, yaw_of(p.orientation)))

    def on_dr(self, m):
        self.dr_t.append(stamp_of(m.header))
        self.dr.append((m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation)))

    def on_slam(self, m):
        if self.done:
            return
        p, c = m.pose.pose, m.pose.covariance
        slam = (p.position.x, p.position.y, yaw_of(p.orientation))
        cov = [[c[0], c[1], c[5]], [c[6], c[7], c[11]], [c[30], c[31], c[35]]]
        self.pending.append((stamp_of(m.header), slam, cov))

    def pair(self):
        """Pair every SLAM pose with the ground truth and dead reckoning interpolated at its stamp."""
        for t, slam, cov in self.pending:
            truth, dr = interpolate(self.truth_t, self.truth, t), interpolate(self.dr_t, self.dr, t)
            if truth is not None and dr is not None:
                self.rows.append((t, *truth, *slam, *dr, cov))

    def on_finished(self, _):
        if self.done:
            return
        self.get_logger().info(f"driver finished; evaluating in {self.settle:.1f} s")
        self.create_timer(self.settle, self.report)

    def report(self):
        if self.done:
            return
        self.done = True
        self.pair()
        n = len(self.rows)
        if n == 0:
            self.get_logger().error("no samples recorded")
            raise SystemExit(1)
        se2 = sum((r[4] - r[1]) ** 2 + (r[5] - r[2]) ** 2 for r in self.rows)
        dr2 = sum((r[7] - r[1]) ** 2 + (r[8] - r[2]) ** 2 for r in self.rows)
        nees = []
        for r in self.rows:
            e = [r[4] - r[1], r[5] - r[2], wrap(r[6] - r[3])]
            nees.append(quad_inv(r[10], e))
        last = self.rows[-1]
        with open(self.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "x_true", "y_true", "yaw_true", "x_slam", "y_slam", "yaw_slam", "x_dr", "y_dr", "yaw_dr",
                        "var_x", "var_y", "var_yaw"])
            for r in self.rows:
                w.writerow([f"{v:.6f}" for v in r[:10]] + [f"{r[10][0][0]:.3e}", f"{r[10][1][1]:.3e}",
                                                           f"{r[10][2][2]:.3e}"])
        lines = [
            f"samples                    {n}",
            f"duration                   {last[0] - self.rows[0][0]:.1f} s",
            f"ATE (RMS position error)   SLAM {math.sqrt(se2 / n):.3f} m   dead reckoning {math.sqrt(dr2 / n):.3f} m",
            f"final position error       SLAM {math.hypot(last[4] - last[1], last[5] - last[2]):.3f} m   "
            f"dead reckoning {math.hypot(last[7] - last[1], last[8] - last[2]):.3f} m",
            f"mean NEES of the SLAM pose {sum(nees) / n:.2f} (consistent: 3)",
            f"written                    {self.output}",
        ]
        print("\n".join(lines), flush=True)
        raise SystemExit(0)


def stamp_of(header):
    return header.stamp.sec + 1e-9 * header.stamp.nanosec


def interpolate(ts, poses, t):
    """Pose at time t, linear in x, y and in the wrapped yaw difference; None outside the recorded interval."""
    k = bisect.bisect_left(ts, t)
    if k < len(ts) and ts[k] == t:
        return poses[k]
    if k == 0 or k == len(ts):
        return None
    t0, t1 = ts[k - 1], ts[k]
    a = (t - t0) / (t1 - t0)
    (x0, y0, h0), (x1, y1, h1) = poses[k - 1], poses[k]
    return (x0 + a * (x1 - x0), y0 + a * (y1 - y0), wrap(h0 + a * wrap(h1 - h0)))


def quad_inv(P, e):
    """e^T P^-1 e for a 3x3 symmetric positive-definite P (Cholesky)."""
    l11 = math.sqrt(P[0][0])
    l21 = P[1][0] / l11
    l31 = P[2][0] / l11
    l22 = math.sqrt(P[1][1] - l21 * l21)
    l32 = (P[2][1] - l31 * l21) / l22
    l33 = math.sqrt(P[2][2] - l31 * l31 - l32 * l32)
    y1 = e[0] / l11
    y2 = (e[1] - l21 * y1) / l22
    y3 = (e[2] - l31 * y1 - l32 * y2) / l33
    return y1 * y1 + y2 * y2 + y3 * y3


def main():
    rclpy.init()
    node = Evaluator()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
