# Hands-On Localization : Pose-Based EKF SLAM (PEKFSLAM) Using ICP Laser Scan Matching

This repository contains the complete source code, simulation files, project report, and demonstration videos for:

**"Pose-Based EKF SLAM (PEKFSLAM) Using ICP Laser Scan Matching"**

### Members
1. [Pravin Oli](mailto:pravin.oli.08@gmail.com)  
2. [Gebrecherkos G.](mailto:chereg2016@gmail.com)  

*Master’s Program in Intelligent Field Robotic Systems (IFRoS)*  
*University of Girona*

---

## Project Overview

This project presents a pose-based Extended Kalman Filter (EKF) SLAM system using laser scan matching with the Iterative Closest Point (ICP) algorithm. The method is designed for mobile robots operating in indoor environments and is validated in both simulation (RViz and Stonefish) and real-world TurtleBot platforms. The system fuses odometry and IMU data for prediction and corrects the robot pose using LIDAR-based ICP registration. Experimental results demonstrate accurate localization and consistent mapping under realistic conditions.
---

## System Components

### Hardware
- **Kobuki TurtleBot 2** – Differential-drive mobile base  
- **uFactory uArm Swift Pro** – 4-DOF robotic manipulator with vacuum gripper  
- **Intel RealSense D435i** – RGB-D camera for object detection  
- **RPLidar A2** – 2D LiDAR for environment sensing  
- **Raspberry Pi 4B** – Onboard processor for real deployment

### Software
- **Ubuntu 20.04 LTS** with **ROS Noetic (ROS 1)**  

- Python scripts  
- Simulation and testing using:
  - [Stonefish Simulator](https://github.com/patrykcieslak/stonefish)  
  - [Stonefish ROS Bridge](https://github.com/patrykcieslak/stonefish_ros)
  
## Demo Video

See inside media folder or click the link below

Drive link  
🔗 [Project Demonstration Video](https://drive.google.com/drive/folders/17vps-_PeFg4AQGtb8tXoSaClggS9_XJu?usp=sharing)

---

## Installation and Execution

Ensure your system is running **Ubuntu 20.04** with **ROS Noetic** installed and sourced.

To install dependencies, clone, build, and launch the project:
```bash
# Step 1: Navigate to your catkin workspace
cd ~/catkin_ws/src

# Step 2: Clone simulation dependencies
git clone https://github.com/patrykcieslak/stonefish.git
git clone https://github.com/patrykcieslak/stonefish_ros.git

# Step 3: Clone this project
git clone https://github.com/Rudip1/localization.git

# Step 4: Build and source the workspace
cd ~/catkin_ws
catkin build
source devel/setup.bash

# Step 5: Run the system

# Launch EKF_Pose_SLAM
roslaunch localization slam_icp.launch







