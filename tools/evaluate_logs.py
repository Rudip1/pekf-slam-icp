#!/usr/bin/env python3
"""Compare the logged SLAM trajectory with ground truth.

Reads data/slam_icp_log.csv, data/ground_truth_log.csv and
data/three_sigma_log.csv (written by slam_icp_node.py on shutdown),
interpolates the ground truth at the SLAM timestamps and prints error and
uncertainty statistics.

Usage: python3 tools/evaluate_logs.py [data_dir]
"""
import csv
import os
import sys

import numpy as np


def load_csv(path):
    """Load a numeric CSV with a header row into a 2D float array."""
    with open(path, newline='') as f:
        rows = list(csv.reader(f))[1:]
    return np.array([[float(v.strip('[]')) for v in row] for row in rows])


def main(data_dir):
    """Print trajectory error and 3-sigma statistics for the logs in ``data_dir``."""
    slam = load_csv(os.path.join(data_dir, 'slam_icp_log.csv'))
    gt = load_csv(os.path.join(data_dir, 'ground_truth_log.csv'))
    sigma = load_csv(os.path.join(data_dir, 'three_sigma_log.csv'))

    t = slam[:, 0]
    gt_x = np.interp(t, gt[:, 0], gt[:, 1])
    gt_y = np.interp(t, gt[:, 0], gt[:, 2])
    gt_th = np.interp(t, gt[:, 0], np.unwrap(gt[:, 3]))

    pos_err = np.hypot(slam[:, 1] - gt_x, slam[:, 2] - gt_y)
    yaw_err = np.arctan2(np.sin(slam[:, 3] - gt_th), np.cos(slam[:, 3] - gt_th))
    path_len = np.sum(np.hypot(np.diff(gt[:, 1]), np.diff(gt[:, 2])))

    print(f"duration            : {t[-1] - t[0]:.1f} s ({len(slam)} SLAM samples)")
    print(f"ground-truth path   : {path_len:.2f} m")
    print(f"position RMSE       : {np.sqrt(np.mean(pos_err ** 2)):.3f} m")
    print(f"position max error  : {pos_err.max():.3f} m")
    print(f"final position error: {pos_err[-1]:.3f} m")
    print(f"yaw RMSE            : {np.degrees(np.sqrt(np.mean(yaw_err ** 2))):.2f} deg")
    print("final 3-sigma [m]   : SLAM x {:.2f}, y {:.2f} | dead reckoning x {:.2f}, y {:.2f}"
          .format(*sigma[-1, 1:]))


if __name__ == '__main__':
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    main(sys.argv[1] if len(sys.argv) > 1 else default_dir)
