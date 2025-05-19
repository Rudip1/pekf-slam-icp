#!/usr/bin/env python3

# -----------------------------
# IMPORTS
# -----------------------------
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Math imports
import numpy as np
import threading
import scipy.linalg
from math import radians, degrees, atan2 , sqrt , pi , floor , cos , sin
from numpy.linalg import eig
import csv

# ROS basics imports
import rospy
import tf
import tf2_ros

# ROS messages Imports
from geometry_msgs.msg import Twist, Quaternion, Point
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# TF transformations Imports
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# Custom modules Imports
from utils_script.ekf_pose_slam import PoseSLAMEKF
from utils_script.pose import Pose3D
from utils_script.helper import *

# -------------------------
# Differential Drive Class
# -------------------------   
class DifferentialDrive:
    """
    Implements a differential-drive robot model with Dead Reckoning + IMU heading correction.
    Uses PoseSLAMEKF for motion prediction and IMU heading update.

    Components:
    - Motion prediction from wheel encoders (joint states).
    - Heading correction using IMU.
    - Visualization: odometry, trajectory, covariance ellipses.

    Dependencies:
    PoseSLAMEKF, tf2, RViz.
    """
    def __init__(self) -> None:
        # -----------------------------
        # Robot state & parameters
        # -----------------------------
        self.xk = np.array([3., -0.78, np.pi/2]).reshape(3, 1)  # Initial robot pose
        self.Pk = np.eye(3) * 0.00001                          # Small initial uncertainty

        # Process noise for prediction
        self.Qk = np.array([[1.8, 0], [0, 1.8]])  # Linear and angular motion noise

        # Compass (IMU) noise
        self.compass_Vk = np.diag([1.0])              # Innovation
        self.compass_Rk = np.diag([0.1**2])           # Measurement

        # Wheel and robot geometry
        self.wheel_radius = 0.035
        self.wheel_base = 0.235

        # Frame names
        self.parent_frame = "world_ned"
        self.child_frame = "turtlebot/base_footprint"
        self.rplidar_frame = "turtlebot/rplidar"  # not used here but preserved

        # Joint names
        self.wheel_name_left = "turtlebot/wheel_left_joint"
        self.wheel_name_right = "turtlebot/wheel_right_joint"

        # Frame names
        #self.parent_frame = "world_ned"
        #self.child_frame  = "turtlebot/kobuki/base_footprint"
        #self.rplidar_frame  = "turtlebot/kobuki/rplidar"
        
        # Wheel joint names (left and right)
        #self.wheel_name_left = "turtlebot/kobuki/wheel_left_joint"
        #self.wheel_name_right = "turtlebot/kobuki/wheel_right_joint"

        # Wheel velocity tracking
        self.left_wheel_velocity = 0
        self.right_wheel_velocity = 0
        self.left_wheel_velo_read = False
        self.right_wheel_velo_read = False

        self.last_time = rospy.Time.now().to_sec()  # Timestamp for delta_t

        # -----------------------------
        # EKF Filter Instance
        # -----------------------------
        self.overlapping_check_th_dis = 1  # dummy value (required param)
        self.pse = PoseSLAMEKF(
            self.xk, self.Pk, self.Qk, self.compass_Rk, self.compass_Vk,
            self.wheel_base, self.wheel_radius, self.overlapping_check_th_dis
        )

        # -----------------------------
        # TF & threading
        # -----------------------------
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_buff = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buff)
        self.mutex = threading.Lock()

        # -----------------------------
        # Visualization state tracking
        # -----------------------------
        self.last_gt_pose = None
        self.gt_speed = 0.0
        self.gt_path = []
        self.initialized = False
        self.gt_theta = 0.0

        # Logging folder
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.csv_slam_log = []  # placeholder: used for log data
        self.csv_gt_log = []

        # -----------------------------
        # ROS Publishers
        # -----------------------------
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)
        self.covariance_markers = rospy.Publisher('/covariance_eigen_markers', MarkerArray, queue_size=10)
        self.trajectory_pub = rospy.Publisher('/slam/trajectory', Marker, queue_size=10)
        self.gt_trajectory_pub = rospy.Publisher('/slam/ground_truth_trajectory', Marker, queue_size=10)

        # Optional:
        self.vel_pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray, queue_size=10)

        # --- ICP/Mapping (Disabled but preserved) ---
        # self.map = []
        # self.scan = []
        # self.scan_cartesian = []
        # self.R_icp = np.diag([0.03**2, 0.03**2, 0.01**2])
        # self.min_scan_th_distance = 0.8
        # self.min_scan_th_angle = np.pi
        # self.max_scan_history = 33
        # self.num_of_scans_to_remove = 4
        # self.max_scans_to_match = 5
        # self.full_map_pub = rospy.Publisher('/slam/map', PointCloud2, queue_size=10)
        # self.full_map_pub_dr = rospy.Publisher('/dr/map', PointCloud2, queue_size=10)
        # self.full_map_pub_gt = rospy.Publisher('/gt/map', PointCloud2, queue_size=10)
        # self.viewpoints_pub = rospy.Publisher("/slam/vis_viewpoints", MarkerArray, queue_size=1)

        # -----------------------------
        # ROS Subscribers
        # -----------------------------
        # rospy.Subscriber("/turtlebot/kobuki/sensors/rplidar", LaserScan, self.check_scan)  # Disable scan matching
        rospy.Subscriber('/turtlebot/joint_states', JointState, self.joint_state_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.velocity_callback)
        rospy.Subscriber('/turtlebot/kobuki/sensors/imu_data', Imu, self.imu_callback)
        rospy.Subscriber('/turtlebot/odom_ground_truth', Odometry, self.ground_truth_callback)

    # -------------------------
    # Data Saving functions
        #1_save_logs_to_csv
    # ------------------------- 
    def save_logs_to_csv(self):
        # Save Dead Reckoning + IMU and Ground Truth logs to CSV files
        dr_path = os.path.join(self.data_dir, 'dr_imu_log.csv')
        gt_path = os.path.join(self.data_dir, 'ground_truth_drimu_log.csv')

        with open(dr_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'dr_x', 'dr_y', 'dr_theta'])
            writer.writerows(self.csv_slam_log)

        with open(gt_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'gt_x', 'gt_y', 'gt_theta'])
            writer.writerows(self.csv_gt_log)

        rospy.loginfo(f"[CSV] Saved DR+IMU log to: {dr_path}")
        rospy.loginfo(f"[CSV] Saved Ground Truth log to: {gt_path}")


    # -------------------------
    # Math utility functions
        #1_wrap_angle
    # -------------------------  
    def wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
       
    # -------------------------
    # ROS Callbacks (main logic) functions
        #1_joint_state_callback
        #2_imu_callback
        #3_velocity_callback
        #4_ground_truth_callback
        #5_check_scan
    # -------------------------
    def joint_state_callback(self, msg):
        """
        Callback for /turtlebot/joint_states topic.

        - Extracts left and right wheel joint velocities.
        - Computes the robot’s linear (v) and angular (w) velocities.
        - Calculates time delta.
        - Performs EKF prediction (motion update) using the wheel encoder input.
        - Publishes estimated odometry and logs the data.
        """

        self.mutex.acquire()  # Ensure thread-safe updates

        # --- STEP 1: Identify left and right wheel velocities from joint names ---
        for i, name in enumerate(msg.name):
            if name == self.wheel_name_left:
                self.left_wheel_velocity = msg.velocity[i]
                self.left_wheel_velo_read = True

            elif name == self.wheel_name_right:
                self.right_wheel_velocity = msg.velocity[i]
                self.right_wheel_velo_read = True

        # --- STEP 2: Wait until both wheel velocities have been read ---
        if self.left_wheel_velo_read and self.right_wheel_velo_read:
            self.left_wheel_velo_read = False
            self.right_wheel_velo_read = False

            # --- STEP 3: Convert joint velocities (rad/s) to linear velocities (m/s) ---
            left_linear_vel = self.left_wheel_velocity * self.wheel_radius
            right_linear_vel = self.right_wheel_velocity * self.wheel_radius

            # --- STEP 4: Compute linear and angular velocity of the robot ---
            self.v = (left_linear_vel + right_linear_vel) / 2.0
            self.w = (left_linear_vel - right_linear_vel) / self.wheel_base

            # --- STEP 5: Calculate time difference since last callback ---
            self.current_time = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs).to_sec()
            self.dt = self.current_time - self.last_time
            self.last_time = self.current_time

            # --- STEP 6: Form control input (delta_x, delta_y=0, delta_theta) ---
            uk = np.array([self.v * self.dt, 0.0, self.w * self.dt]).reshape(3, 1)

            # --- STEP 7: EKF motion update (Prediction step) ---
            self.xk, self.Pk = self.pse.Prediction(self.xk, self.Pk, uk, self.dt)

            # --- STEP 8: Publish odometry and log ---
            self.publish_odometry(msg)

        self.mutex.release()

        # --- Optional: update RViz trajectory after prediction ---
        self.publish_trajectory()

        if hasattr(self, 'gt_path') and len(self.gt_path) > 0:
            self.publish_ground_truth_trajectory()
      
    def imu_callback(self, msg):
        """
        Callback for /turtlebot/kobuki/sensors/imu_data topic.

        - Extracts yaw (heading) angle from IMU orientation.
        - Updates the EKF state with the new heading measurement.
        """

        self.mutex.acquire()  # Lock shared variables for thread safety

        # --- STEP 1: Extract quaternion and convert to Euler angles ---
        orientation = msg.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        # --- STEP 2: Wrap yaw to [-pi, pi] ---
        yaw = self.wrap_angle(yaw)

        # --- STEP 3: Format yaw as a 1x1 column vector ---
        yaw_measurement = np.array([yaw]).reshape(1, 1)

        # --- STEP 4: EKF heading update using the compass (IMU yaw) ---
        self.xk, self.Pk = self.pse.heading_update(self.xk, self.Pk, yaw_measurement)

        self.mutex.release()

    def velocity_callback(self, msg):
        """
        Callback for /cmd_vel topic.

        Converts Twist message (linear.x, angular.z) into individual wheel velocities,
        then publishes them to the robot's wheel controllers.

        Used when controlling the robot with rqt_robot_steering or other ROS tools.
        """

        # --- Extract desired linear and angular velocity ---
        lin_vel = msg.linear.x        # Forward speed in m/s
        ang_vel = msg.angular.z       # Rotational speed in rad/s

        # --- Compute linear velocity for each wheel (m/s) ---
        left_linear_vel  = lin_vel - (ang_vel * self.wheel_base / 2.0)
        right_linear_vel = lin_vel + (ang_vel * self.wheel_base / 2.0)

        # --- Convert linear velocity to angular velocity (rad/s) ---
        left_wheel_velocity  = left_linear_vel / self.wheel_radius
        right_wheel_velocity = right_linear_vel / self.wheel_radius

        # --- Package as Float64MultiArray and publish to wheel command topic ---
        wheel_vel = Float64MultiArray()
        wheel_vel.data = [left_wheel_velocity, right_wheel_velocity]
        self.vel_pub.publish(wheel_vel)

        # Optional debug
        rospy.loginfo_once("[vel_callback] Received /cmd_vel and published wheel velocities.")

    def ground_truth_callback(self, msg):
        """
        Receives ground truth Odometry and initializes EKF + tracks GT path.
        Also logs ground truth data to CSV with timestamps.
        """
        # Extract ground truth pose
        gt_x = msg.pose.pose.position.x
        gt_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        (_, _, gt_theta) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.gt_theta = gt_theta

        # Save full GT pose for CSV
        current_time = rospy.Time.now().to_sec()
        self.csv_gt_log.append([current_time, gt_x, gt_y, gt_theta])

        # Initialize EKF once using GT pose
        if not hasattr(self, 'initialized') or not self.initialized:
            self.xk = np.array([gt_x, gt_y, gt_theta]).reshape(3, 1)
            rospy.loginfo(f"EKF initialized from ground truth: [{gt_x:.2f}, {gt_y:.2f}, {degrees(gt_theta):.1f}°]")
            self.initialized = True

        # Always save ground truth pose for trajectory
        if not hasattr(self, 'gt_path'):
            self.gt_path = []
        self.gt_path.append((gt_x, gt_y))

    def publish_odometry(self, msg):
        """
        Publishes odometry and TF transform using the current EKF-estimated state.

        Args:
            msg (JointState): The incoming joint state message (used for timestamp).
        """

        # --- STEP 1: Create Odometry message ---
        odom = Odometry()

        # Extract timestamp from joint state
        current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)

        # Convert current yaw (theta) to quaternion for orientation
        theta = float(self.xk[-1])
        q = quaternion_from_euler(0, 0, theta)

        # --- STEP 2: Fill in pose and twist ---
        odom.header.stamp = current_time
        odom.header.frame_id = self.parent_frame       # Usually "world_ned"
        odom.child_frame_id = self.child_frame         # Usually "base_footprint"

        odom.pose.pose.position.x = float(self.xk[-3])
        odom.pose.pose.position.y = float(self.xk[-2])
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w

        # --- STEP 3: Fill covariance matrix (only pose part is non-zero) ---
        covar = [   self.Pk[-3,-3], self.Pk[-3,-2], 0.0, 0.0, 0.0, self.Pk[-3,-1],
                    self.Pk[-2,-3], self.Pk[-2,-2], 0.0, 0.0, 0.0, self.Pk[-2,-1],
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    self.Pk[-1,-3], self.Pk[-1,-2], 0.0, 0.0, 0.0, self.Pk[-1,-1] ]
        
        odom.pose.covariance = covar

        # --- STEP 4: Publish odometry message ---
        self.odom_pub.publish(odom)

        # --- STEP 5: Publish TF for visualization and TF tree ---
        self.tf_broadcaster.sendTransform(
            (self.xk[-3], self.xk[-2], 0.0),  # translation
            q,                               # quaternion
            rospy.Time.now(),                # time
            self.child_frame,                # child
            self.parent_frame                # parent
        )

        # Warn if ground truth hasn't been used yet for initialization
        if not self.initialized:
            rospy.logwarn_once("EKF not initialized from ground truth yet; publishing default pose.")
    
    # -------------------------
    # Visualization (RViz markers) functions
        #1_publish_trajectory
        #2_publish_ground_truth_trajectory
        #3_publish_viewpoints
        #4_publish_covariance_marker
    # -------------------------
    def publish_trajectory(self):
        """
        Publish the full trajectory as a line strip in RViz.
        """
        path_marker = Marker()
        path_marker.header.frame_id = "world_ned"
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.id = 999  # unique ID for the trajectory marker

        path_marker.scale.x = 0.0175  # Line width
        path_marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)  # Green, opaque

        # Add each pose (x, y) to the path
        for i in range(0, len(self.xk), 3):
            point = Point()
            point.x = self.xk[i]
            point.y = self.xk[i+1]
            point.z = 0.0
            path_marker.points.append(point)

        self.trajectory_pub.publish(path_marker)
    
    def publish_ground_truth_trajectory(self):
        """
        Publish the ground truth trajectory as a line strip in RViz.
        """
        path_marker = Marker()
        path_marker.header.frame_id = "world_ned"
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.id = 1000  # unique ID

        path_marker.scale.x = 0.0175 # Line width
        path_marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)  # Red, semi-transparent

        for (x, y) in self.gt_path:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            path_marker.points.append(point)

        self.gt_trajectory_pub.publish(path_marker)  
    
    def publish_covariance_marker(self):
        """
        Publishes covariance ellipses at the robot's current and past estimated poses.
        Visualizes uncertainty using eigenvalues/eigenvectors of the 2x2 position covariance.
        """
        marker_array = MarkerArray()
        id = 0
        markers = []

        # Go through each pose (x, y, theta) in state vector
        for j in range(0, len(self.xk), 3):
            # Extract 2x2 covariance (for x, y)
            covariance_xy = self.Pk[j:j+2, j:j+2]
            eigenvalues, eigenvectors = eig(covariance_xy)

            marker = Marker()
            marker.header.frame_id = "world_ned"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Default ellipse size
            marker_scale = [0.1, 0.1, 0.00001]

            # If eigenvalues are valid, scale ellipse axes
            if not np.any(np.isnan(eigenvalues)) and not np.any(np.isinf(eigenvalues)) and np.all(eigenvalues >= 0):
                marker_scale = [
                    2.4 * np.sqrt(eigenvalues[0]),
                    2.4 * np.sqrt(eigenvalues[1]),
                    0.0001  # flat in z-axis
                ]

            # Bright cyan marker (good contrast for dark backgrounds)
            marker_color = [0.0, 1.0, 1.0, 0.9]  # RGBA

            # Assign marker details
            id += 1
            marker.id = id
            marker.scale.x = marker_scale[0]
            marker.scale.y = marker_scale[1]
            marker.scale.z = 0.1
            marker.color.r = marker_color[0]
            marker.color.g = marker_color[1]
            marker.color.b = marker_color[2]
            marker.color.a = marker_color[3]
            marker.pose.position.x = float(self.xk[j])
            marker.pose.position.y = float(self.xk[j+1])
            marker.pose.position.z = 0.0

            # Orient the ellipse using eigenvectors
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            quat = tf.transformations.quaternion_from_euler(0, 0, angle)
            quat /= np.linalg.norm(quat)

            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]

            markers.append(marker)

        marker_array.markers.extend(markers)
        self.covariance_markers.publish(marker_array)


# -------------------------
# Main Entry Point
# -------------------------
if __name__ == '__main__': 
    rospy.init_node('differential_robot_node')  
    # Creates the ROS node (must be called before any ROS communication)

    diff_drive = DifferentialDrive()  
    # Instantiates your robot model and sets up all publishers, subscribers, and state

    rospy.on_shutdown(diff_drive.save_logs_to_csv)  
    # Ensures that when you Ctrl+C or exit, your ground truth and DR+IMU logs are saved to CSV

    rospy.spin()  
    # Keeps the node running and responding to callbacks



