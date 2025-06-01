#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------
# Resolve path to data folder
# --------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
error_file = os.path.join(data_dir, 'three_sigma_log.csv')

# --------------------------------------
# Load CSV and check existence
# --------------------------------------
if not os.path.exists(error_file):
    raise FileNotFoundError(f"Missing CSV file: {error_file}")

df = pd.read_csv(error_file)
df = pd.read_csv(error_file)

# Normalize time to start at 0
df['time'] -= df['time'].iloc[0]


# --------------------------------------
# Plotting EKF SLAM and DR Uncertainties
# --------------------------------------
plt.figure(figsize=(10, 6))

# Plot EKF SLAM 3-sigma uncertainty (x and y)
plt.plot(df['time'], df['slam_3sigma_x'], color='red', label='EKF SLAM σₓ')
plt.plot(df['time'], df['slam_3sigma_y'], color='green', label='EKF SLAM σᵧ')

# Plot Dead Reckoning 3-sigma uncertainty (x and y)
plt.plot(df['time'], df['dr_3sigma_x'], color='blue', label='Dead Reckoning σₓ')
plt.plot(df['time'], df['dr_3sigma_y'], color='black', label='Dead Reckoning σᵧ')

# --------------------------------------
# Customize plot appearance
# --------------------------------------
plt.title('Robot Pose Uncertainty Over Time')
plt.xlabel('Time (s)')
plt.ylabel(r'$3\sigma$ Deviation (m)')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# --------------------------------------
# Display the plot
# --------------------------------------
plt.show()
