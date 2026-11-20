#!/usr/bin/env python3
"""Plot a run written by evaluate_run.py and print its metrics.

    python3 tools/plot_run.py docs/results/two_laps_imu.csv docs/results/two_laps_imu.png

Left: ground truth, SLAM and dead-reckoning trajectories over the office walls. Right: SLAM position error over
time with its 3-sigma bound from the filter covariance, and the dead-reckoning error.
"""
import csv
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The office walls, as in pekf_slam_icp_gazebo/scripts/make_office_world.py.
WALLS = [(-8, -6, 8, -6), (8, -6, 8, 6), (8, 6, -8, 6), (-8, 6, -8, -6),
         (-3, -6, -3, -4.9), (-3, -3.1, -3, 3.7), (-3, 5.3, -3, 6),
         (3, -6, 3, -4.9), (3, -3.3, 3, 3.4), (3, 5.0, 3, 6),
         (-1, -2.5, 1, -2.5), (0, 2.5, 1, 3.3), (-4.6, 0.5, -3.8, 1.3), (3.8, -2, 4.6, -1.2),
         (-7.5, -1, -7.5, 1), (7.4, 2, 7.4, 3.5)]


def load(path):
    with open(path) as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return {k: [r[k] for r in rows] for k in rows[0]}


def main(src, dst):
    d = load(src)
    t0 = d["t"][0]
    t = [v - t0 for v in d["t"]]
    e_slam = [math.hypot(a - b, c - e) for a, b, c, e in zip(d["x_slam"], d["x_true"], d["y_slam"], d["y_true"])]
    e_dr = [math.hypot(a - b, c - e) for a, b, c, e in zip(d["x_dr"], d["x_true"], d["y_dr"], d["y_true"])]
    bound = [3 * math.sqrt(vx + vy) for vx, vy in zip(d["var_x"], d["var_y"])]
    n = len(t)
    print(f"duration {t[-1]:.1f} s, {n} samples")
    print(f"ATE: SLAM {math.sqrt(sum(e * e for e in e_slam) / n):.3f} m, "
          f"dead reckoning {math.sqrt(sum(e * e for e in e_dr) / n):.3f} m")
    print(f"max SLAM error {max(e_slam):.3f} m; final: SLAM {e_slam[-1]:.3f} m, dead reckoning {e_dr[-1]:.3f} m")
    # Total rotation seen by the encoders against the true rotation: a ratio other than 1 is a systematic odometry
    # error (an effective wheel separation different from the nominal one), not random slip.
    turn = lambda yaw: sum(abs(math.atan2(math.sin(b - a), math.cos(b - a))) for a, b in zip(yaw, yaw[1:]))
    print(f"encoder rotation / true rotation over the run: {turn(d['yaw_dr']) / turn(d['yaw_true']):.3f}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax_, ay_, bx_, by_ in WALLS:
        a.plot([ax_, bx_], [ay_, by_], color="0.6", lw=3, solid_capstyle="round")
    a.plot(d["x_true"], d["y_true"], "k--", lw=1, label="ground truth")
    a.plot(d["x_dr"], d["y_dr"], color="#d62728", lw=1, label="dead reckoning (wheel encoders)")
    a.plot(d["x_slam"], d["y_slam"], color="#1f77b4", lw=1.3, label="pose-based EKF SLAM")
    a.set_aspect("equal"); a.grid(alpha=0.3); a.legend(fontsize=8, loc="lower right")
    a.set_title("trajectories in the simulated office")
    b.plot(t, e_dr, color="#d62728", lw=1, label="dead-reckoning position error")
    b.plot(t, e_slam, color="#1f77b4", lw=1.2, label="SLAM position error")
    b.plot(t, bound, color="#1f77b4", ls=":", lw=1, label="SLAM 3σ bound (filter covariance)")
    b.set_xlabel("time [s]"); b.set_ylabel("position error [m]"); b.set_yscale("log"); b.grid(alpha=0.3, which="both")
    b.legend(fontsize=8); b.set_title("error against ground truth")
    fig.tight_layout()
    fig.savefig(dst, dpi=110)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
