#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt

# Resolve path to data/ folder (one level up from this script)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')

slam_file = os.path.join(data_dir, 'slam_icp_log.csv')
gt_file = os.path.join(data_dir, 'ground_truth_log.csv')

# Check file existence
if not os.path.exists(slam_file) or not os.path.exists(gt_file):
    raise FileNotFoundError("CSV files not found in ../data/. Please run SLAM and shutdown cleanly to generate them.")

# Load CSVs
slam_df = pd.read_csv(slam_file)
gt_df = pd.read_csv(gt_file)

# Normalize time
slam_df['time'] -= slam_df['time'].iloc[0]
gt_df['time'] -= gt_df['time'].iloc[0]

# Solid	'-'
# Dashed	'--'
# Dotted	':'
# Dash-dot	'-.'
# Plot
plt.figure(figsize=(10, 7))

# ---- X vs Time ----
plt.subplot(3, 1, 1)
plt.plot(slam_df['time'], slam_df['slam_x'], label='slam x', color='green', linestyle='--', linewidth=2.0)
plt.plot(gt_df['time'], gt_df['gt_x'], label='ground truth', color='red', linestyle=':', linewidth=2.0)
plt.title('x vs time')
plt.ylabel('x [m]')
plt.legend()
plt.grid()

# ---- Y vs Time ----
plt.subplot(3, 1, 2)
plt.plot(slam_df['time'], slam_df['slam_y'], label='slam y', color='green', linestyle='--', linewidth=2.0)
plt.plot(gt_df['time'], gt_df['gt_y'], label='ground truth', color='red', linestyle=':', linewidth=2.0)
plt.title('y vs time')
plt.ylabel('y [m]')
plt.legend()
plt.grid()

# ---- Theta vs Time ----
plt.subplot(3, 1, 3)
plt.plot(slam_df['time'], slam_df['slam_theta'], label='slam theta', color='green', linestyle='--', linewidth=2.0)
plt.plot(gt_df['time'], gt_df['gt_theta'], label='ground truth', color='red', linestyle=':', linewidth=2.0)
plt.title('theta vs time')
plt.xlabel('time [s]')
plt.ylabel('theta [rad]')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
