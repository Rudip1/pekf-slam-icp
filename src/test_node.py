#!/usr/bin/env python3

# -----------------------------
# IMPORTS
# -----------------------------
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Math imports

import csv
import numpy as np
import threading
import scipy.linalg
from math import radians, degrees
from numpy.linalg import eig

# ROS basics imports
import rospy
import tf
import tf2_ros

# ROS messages Imports
from geometry_msgs.msg import Twist, Quaternion, Point
from sensor_msgs.msg import JointState, Imu, LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs import point_cloud2

# TF transformations Imports
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

# Custom modules Imports
from utils_script.ekf_pose_slam import PoseSLAMEKF
from utils_script.icp import ICP
from utils_script.pose import Pose3D
from utils_script.helper import *

# -------------------------
# Differential Drive Class
# -------------------------
class DifferentialDrive:
    """
    Implements a differential-drive robot model with Pose-based EKF SLAM.

    Core components:
    - Motion prediction from wheel encoders (joint states).
    - Heading correction using IMU.
    - Scan matching with ICP for map updates.
    - Visualization: odometry, trajectory, covariance ellipses, map points.
    - Publishes wheel velocities based on /cmd_vel.

    Dependencies:
    PoseSLAMEKF, ICP, tf2, RViz.
    """
    def __init__(self) -> None:
        # -----------------------------
        # Robot state & parameters
        # -----------------------------
        self.xk      = np.array([0, 0, 0]).reshape(3, 1) # Initialize the state of the robot
        self.xk      = np.array([3., -0.78, np.pi/2]).reshape(3, 1) # Initialize the state of the robot
        self.Pk      = np.eye(3)*0.00001# Initialize the covariance of the state
        self.Qk = np.array([[1.8,0 ],[0,1.8],]) # Initialize the covariance of the process noise  
        #self.Qk     = np.diag([0.005, 0.005, radians(0.2)])         # Process noise (motion model uncertainty)
        self.R_icp = np.diag([0.03**2, 0.03**2, 0.01**2])   # ICP registration noise

        self.compass_Vk = np.diag([1.0])                            # Compass measurement uncertainty
        self.compass_Rk = np.diag([0.1**2])                         # Compass measurement covariance matrix noise

        # #self.xk      = np.array([0, 0, 0]).reshape(3, 1)           # Initialize the state of the robot
        # self.xk     = np.array([3.0, -0.78, np.pi/2]).reshape(3, 1)  # Initialize the state of the robot
        # self.Pk     = np.eye(3)*0.00001                             # Initialize the covariance of the state
        # self.Qk     = np.diag([0.005, 0.005, radians(0.2)])         # Process noise (motion model uncertainty)
        # self.R_icp = np.diag([0.15**2, 0.15**2, radians(7)**2])


        # self.compass_Vk = np.diag([1.0])                            # Compass measurement uncertainty
        # self.compass_Rk = np.diag([0.01])                           # Compass measurement covariance matrix noise
        
        # Wheel properties, Robot geometry (meters)
        self.wheel_radius = 0.035
        self.wheel_base = 0.235

         # Initialize the topics
        # Frame names
        self.parent_frame  = "world_ned"
        self.child_frame  = "turtlebot/base_footprint"
        self.rplidar_frame = "turtlebot/rplidar"

        # Wheel joint names (left and right)
        self.wheel_name_left  = "turtlebot/wheel_left_joint"
        self.wheel_name_right = "turtlebot/wheel_right_joint"

        # Frame names
        #self.parent_frame = "world_ned"
        #self.child_frame  = "turtlebot/kobuki/base_footprint"
        #self.rplidar_frame  = "turtlebot/kobuki/rplidar"
        
        # Wheel joint names (left and right)
        #self.wheel_name_left = "turtlebot/kobuki/wheel_left_joint"
        #self.wheel_name_right = "turtlebot/kobuki/wheel_right_joint"

        self.left_wheel_velocity   = 0
        self.right_wheel_velocity  = 0
        self.left_wheel_velo_read  = False
        self.right_wheel_velo_read = False

        # Time tracking for delta_t
        self.last_time = rospy.Time.now().to_sec()

        # scan matching variables 
        self.min_scan_th_distance = 0.8  # minimum scan taking distance  
        self.min_scan_th_angle = np.pi # take scan angle thershold
        self.overlapping_check_th_dis = 1 # ovrlapping checking distance thershold 
        self.max_scan_history = 33 # maximum amount of scan history to store 
        self.num_of_scans_to_remove = 4 # number of scans to remove from the history
        self.max_scans_to_match = 5 # maximum number of scans to match

        # # Scan matching thresholds & settings
        # self.min_scan_th_distance = 0.4  # minimum scan taking distance  
        # self.min_scan_th_angle = np.pi/3 # take scan angle thershold
        # self.overlapping_check_th_dis = 1 # ovrlapping checking distance thershold 
        # self.max_scan_history = 200 # maximum amount of scan history to store 
        # self.num_of_scans_to_remove = 4 # number of scans to remove from the history
        # self.max_scans_to_match = 6 # maximum number of scans to match


        # EKF SLAM instance (create object of pose based slam )
        """Creates the SLAM filter with robots parameters.
        PoseSLAMEKF is the EKF to use in prediction & update.
        Stores motion model and noise models.
        """
        self.pse = PoseSLAMEKF(
            self.xk, self.Pk, self.Qk, self.compass_Rk, self.compass_Vk,
            self.wheel_base, self.wheel_radius, self.overlapping_check_th_dis
        )

        # TF & threading
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_buff = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buff)
        self.mutex = threading.Lock() # mutex for threading safety

        # State trackers
        self.last_gt_pose = None    # To store previous ground truth pose (x, y)
        self.gt_speed = 0.0         # To store computed ground truth speed (updated every time)
        self.gt_path = []           # to store ground truth trajectory
        self.initialized = False    # to track EKF initialization

        # Initialize storage for scans and maps
        self.map = []   # Initialize the map
        self.scan = []  # Initialize the scan
        self.scan_cartesian = []
        self.gt_theta = 0.0

        # Automatically get path to 'data' folder inside the same package
        # self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        # if not os.path.exists(self.data_dir):
        #     os.makedirs(self.data_dir)

        # Automatically get path to 'hol/data' folder (one level above src/)
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Initialize the csv log for slam and ground truth
        self.csv_slam_log = [] #Initialize the csv log for slam
        self.csv_gt_log = [] #Initialize the csv log for ground truth

        # -----------------------------
        # ROS Publishers
        # -----------------------------
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10) # Odometry publisher
        self.covariance_markers = rospy.Publisher('/covariance_eigen_markers', MarkerArray, queue_size=10) # Covariance markers publisher
        self.viewpoints_pub = rospy.Publisher("/slam/vis_viewpoints", MarkerArray, queue_size=1) # Viewpoints publisher
        self.full_map_pub = rospy.Publisher('/slam/map', PointCloud2, queue_size=10) # EKF slam Map publisher

        self.full_map_pub_dr = rospy.Publisher('/dr/map', PointCloud2, queue_size=10)  # DeadReckoning map publisher
        self.full_map_pub_gt = rospy.Publisher('/gt/map', PointCloud2, queue_size=10)  # Ground truth map publisher

        self.vel_pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray, queue_size=10) # joint1 velocity publisher
        self.trajectory_pub = rospy.Publisher('/slam/trajectory', Marker, queue_size=10) # EKF trajectory publisher
        self.gt_trajectory_pub = rospy.Publisher('/slam/ground_truth_trajectory', Marker, queue_size=10) # ground truth trajectory publisher

        # -----------------------------
        # ROS Subscribers
        # -----------------------------
        rospy.Subscriber("/turtlebot/kobuki/sensors/rplidar", LaserScan, self.check_scan)      # RPLidar subscriber
        rospy.Subscriber('/turtlebot/joint_states', JointState, self.joint_state_callback)     # Joint state subscriber
        rospy.Subscriber('/cmd_vel', Twist, self.velocity_callback)                            # Velocity subscriber
        rospy.Subscriber('/turtlebot/kobuki/sensors/imu_data', Imu, self.imu_callback)         # Imu subscriber
        rospy.Subscriber('/turtlebot/odom_ground_truth', Odometry, self.ground_truth_callback) # Ground truth subscriber
   
    # -------------------------
    # Data Saving functions
        #1_save_logs_to_csv
    # ------------------------- 
    def save_logs_to_csv(self):
        # Save SLAM and Ground Truth logs to CSV files
        slam_path = os.path.join(self.data_dir, 'slam_icp_log.csv')
        gt_path = os.path.join(self.data_dir, 'ground_truth_log.csv')

        with open(slam_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'slam_x', 'slam_y', 'slam_theta'])
            writer.writerows(self.csv_slam_log)

        with open(gt_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'gt_x', 'gt_y', 'gt_theta'])
            writer.writerows(self.csv_gt_log)

        rospy.loginfo(f"[CSV] Saved SLAM to: {slam_path}")
        rospy.loginfo(f"[CSV] Saved GT   to: {gt_path}")

    # -------------------------
    # Math utility functions
        #1_wrap_angle
    # -------------------------  
    def wrap_angle(self, angle):
        """this function wraps the angle between -pi and pi

        :param angle: the angle to be wrapped
        :type angle: float

        :return: the wrapped angle
        :rtype: float
        """
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    
    # -------------------------
    # TF utility functions
        #1_transform_cloud
    # -------------------------
    def transform_cloud(self, target_frame, source_frame, scan):
        """
        Transforms LaserScan data from the source frame to the target frame.

        This function:
        - Converts a LaserScan to a PointCloud2 message.
        - Looks up the transform between source_frame and target_frame.
        - Applies the transform to the point cloud.
        - Returns a list of (x, y) points in the target frame.

        Args:
            target_frame (str): The desired target frame (e.g., 'turtlebot/base_footprint').
            source_frame (str): The frame in which the scan was originally captured (e.g., 'turtlebot/rplidar').
            scan (LaserScan): The incoming LaserScan message to transform.

        Returns:
            list of [x, y] points (list of lists), or None if the transform failed.

        Notes:
            - We use rospy.Time(0) for the latest available transform to be robust.
            - If no transform is found within 1 second, logs an error and returns None.
        """
        try:
            # Project LaserScan into a PointCloud2 (requires laser_geometry)
            projector = LaserProjection()
            cloud = projector.projectLaser(scan)

            # Ensure the transform is available before proceeding
            self.tf_buff.can_transform(target_frame, source_frame, scan.header.stamp, rospy.Duration(1.0))

            # Lookup the transform: rospy.Time(0) = get the latest available transform
            transform = self.tf_buff.lookup_transform(target_frame, source_frame, rospy.Time(0))

            # Apply the transform to the point cloud
            transformed_cloud = do_transform_cloud(cloud, transform)

            # Extract (x, y) points from the transformed cloud
            xy_points = [
                [point[0], point[1]]
                for point in point_cloud2.read_points(transformed_cloud, field_names=("x", "y"), skip_nans=True)
            ]

            return xy_points

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"[transform_cloud] Transform failed from {source_frame} to {target_frame}: {e}")
            return None
       
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

        - Reads the left and right wheel joint velocities.
        - Computes the robot's linear and angular velocities.
        - Performs EKF prediction (motion update).
        - Publishes updated odometry.

        Args:
            msg (JointState): ROS JointState message containing wheel velocities.
        """
        self.mutex.acquire()
        # Detect which wheel this message refers to and store the velocity
        if msg.name[0] == self.wheel_name_left:
            self.left_wheel_velocity = msg.velocity[0]
            self.left_wheel_velo_read = True

        elif msg.name[0] == self.wheel_name_right:
            self.right_wheel_velocity = msg.velocity[0]
            self.right_wheel_velo_read = True

        # Once both wheels have been updated, process odometry
        if self.left_wheel_velo_read and self.right_wheel_velo_read:
            # Reset flags
            self.left_wheel_velo_read = False
            self.right_wheel_velo_read = False

            # Convert wheel velocities (rad/s) to linear velocities (m/s)
            left_linear_vel = self.left_wheel_velocity * self.wheel_radius
            right_linear_vel = self.right_wheel_velocity * self.wheel_radius

            # Calculate linear (v) and angular (w) velocities of the robot
            self.v = (left_linear_vel + right_linear_vel) / 2.0
            self.w = (left_linear_vel - right_linear_vel) / self.wheel_base

            # Compute delta time
            self.current_time = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs).to_sec()
            self.dt = self.current_time - self.last_time
            self.last_time = self.current_time

            # Build control input (dx, dy=0, dtheta)
            uk = np.array([self.v*self.dt, 0, self.w*self.dt]).reshape(3, 1)
   
            # EKF Prediction step (motion update)
            self.xk , self.Pk = self.pse.Prediction( self.xk , self.Pk , uk ,self.dt)

            # Publish updated odometry
            self.publish_odometry(msg)

        self.mutex.release()

        
    def imu_callback(self, msg):
        '''This function reads the IMU data and updates the heading of the robot    
        Args:
            msg (Imu): The IMU message
            '''
        self.mutex.acquire()
        orientation = msg.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.yaw = np.array([yaw]).reshape(1, 1)
        self.heading_updae = True
        self.xk ,self.Pk = self.pse.heading_update(self.xk , self.Pk , self.yaw)
        self.mutex.release()

    def velocity_callback(self, msg):
        lin_vel = msg.linear.x
        ang_vel = msg.angular.z

        # print("linear and angular ", lin_vel , ang_vel )
        left_linear_vel   = lin_vel  - (ang_vel*self.wheel_base/2)
        right_linear_vel = lin_vel   +  (ang_vel*self.wheel_base/2)
 
        left_wheel_velocity  = left_linear_vel / self.wheel_radius
        right_wheel_velocity = right_linear_vel / self.wheel_radius
        
        # print("left_wheel_velocity",left_wheel_velocity , right_wheel_velocity)
        
        wheel_vel = Float64MultiArray()
        wheel_vel.data = [left_wheel_velocity, right_wheel_velocity]
        self.vel_pub.publish(wheel_vel)
    
    # def ground_truth_callback(self, msg):
    #     """
    #     Receives ground truth Odometry and initializes EKF + tracks GT path.
    #     """
    #     # Extract ground truth pose
    #     gt_x = msg.pose.pose.position.x
    #     gt_y = msg.pose.pose.position.y
    #     q = msg.pose.pose.orientation
    #     (_, _, gt_theta) = euler_from_quaternion([q.x, q.y, q.z, q.w])
    #     self.gt_theta = gt_theta

    #     # Initialize EKF once using GT pose
    #     if not hasattr(self, 'initialized') or not self.initialized:
    #         self.xk = np.array([gt_x, gt_y, gt_theta]).reshape(3, 1)
    #         rospy.loginfo(f"EKF initialized from ground truth: [{gt_x:.2f}, {gt_y:.2f}, {degrees(gt_theta):.1f}°]")
    #         self.initialized = True

    #     # Always save ground truth pose for trajectory
    #     if not hasattr(self, 'gt_path'):
    #         self.gt_path = []
    #     self.gt_path.append((gt_x, gt_y))

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

    
    def check_scan(self, scan):
        '''This function checks the scan data and updates the map
        Args:
            scan (LaserScan): The scan data
            '''
        self.mutex.acquire()
        scan = self.transform_cloud(self.child_frame , self.rplidar_frame , scan)
        # Convert laser scan data to x, y coordinates in robot frame 
        self.scan_cartesian = np.array(scan)
    
        if(len(self.map) == 0 and len(self.scan_cartesian)):
            self.publish_covariance_marker()
            self.scan_world = scan_to_world(self.scan_cartesian, self.xk[-3:])
            self.scan.append(self.scan_cartesian)
            self.map.append(self.scan_world)    
            self.xk , self.Pk = self.pse.Add_New_Pose(self.xk, self.Pk)
        elif(len(self.scan_cartesian) and check_scan_threshold(self.xk,self.min_scan_th_distance, self.min_scan_th_angle)):
            self.publish_covariance_marker()
            self.scan_world = scan_to_world(self.scan_cartesian  , self.xk[-3:])
            self.map.append(self.scan_world)
            self.scan.append(self.scan_cartesian)
            self.xk , self.Pk = self.pse.Add_New_Pose(self.xk, self.Pk)

            # check if the scan is overlapping with the previous scans
            Ho = self.pse.OverlappingScan(self.xk , self.overlapping_check_th_dis , self.max_scans_to_match)            
            zp = np.zeros((0,1)) # predicted scan
            Rp = np.zeros((0,0)) # predicted scan covariance
            Hp = []
            i =0 
            for j in Ho:
                jXk = self.pse.hfj(self.xk, j)
                jPk = self.pse.jPk(self.xk, self.Pk ,j)
                matched_scan = self.scan[j]
                current_scan = self.scan[-1]
                xk  = self.xk[-3:].reshape(3,1)
                # ICP Registration 
                zr  = ICP(matched_scan, current_scan, jXk)
                matched_pose = self.xk[j:j+3,:].reshape((3,1))
                Rr = self.R_icp
                isCompatibile = self.pse.ICNN(jXk, jPk , zr, Rr )
                if(isCompatibile):
                    zp = np.block([[zp],[zr]])
                    Rp = scipy.linalg.block_diag(Rp, Rr)
                    Hp.append(j)
            if(len(Hp)>0):
                zk ,Rk, Hk,Vk = self.pse.ObservationMatrix(self.xk , Hp,zp,Rp )
                self.xk , self.Pk = self.pse.Update(self.xk, self.Pk, Hk, Vk, zk, Rk,Hp)

            # remove the oldest scan
            if(len(self.map) > self.max_scan_history):
                  
                self.xk , self.Pk = self.pse.remove_pose(self.xk, self.Pk , self.num_of_scans_to_remove) 
               
                indices = [2*i+1 for i in range(self.num_of_scans_to_remove)]
             
                for i in sorted(indices, reverse=True):
                    self.scan.pop(i)
                    self.map.pop(i)
                # for i in range(self.num_of_scans_to_remove ):
                #     print("remove")
                #     print("scan", len(self.scan))
                #     print("map", len(self.map))
                #     self.scan.pop(0)
                #     self.map.pop(0)
                #     print("scan", len(self.scan))
                #     print("map", len(self.map))

        # Always update RViz
        if len(self.scan) > 0:
            self.publish_viewpoints()
            self.create_map()
            self.publish_trajectory()
            self.publish_ground_truth_trajectory()      
            
        self.mutex.release() 

        # Save ICP SLAM and Ground Truth data at every scan
        current_time = rospy.Time.now().to_sec()

        # Extract SLAM pose
        slam_x = float(self.xk[-3])
        slam_y = float(self.xk[-2])
        slam_theta = float(self.xk[-1])

        # Append to SLAM log
        self.csv_slam_log.append([current_time, slam_x, slam_y, slam_theta])

        # # Extract latest ground truth if available
        # if hasattr(self, 'gt_path') and len(self.gt_path) > 0:
        #     gt_x, gt_y = self.gt_path[-1]
        #     gt_theta = float(self.xk[2])  # You can replace this with a stored GT theta if tracked separately
        #     #self.csv_gt_log.append([current_time, gt_x, gt_y, gt_theta])
        #     self.csv_gt_log.append([current_time, gt_x, gt_y, self.gt_theta])

    # -------------------------
    # Odometry + TF functions
        #1_publish_odometry
    # -------------------------
    def publish_odometry(self ,msg):
        """
        Publishes the odometry message
        """
        '''This function publishes the odometry message
        Args:
            msg (JointState): The joint state message
        '''
       
        odom = Odometry()
        
        current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
     
        theta = self.xk[-1].copy()
        q = quaternion_from_euler(0, 0, float(theta))
        
        covar = [   self.Pk[-3,-3], self.Pk[-3,-2], 0.0, 0.0, 0.0, self.Pk[-3,-1],
                    self.Pk[-2,-3], self.Pk[-2,-2], 0.0, 0.0, 0.0, self.Pk[-2,-1],  
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    self.Pk[-1,-3], self.Pk[-1,-2], 0.0, 0.0, 0.0, self.Pk[-1,-1]]

        odom.header.stamp = current_time
        odom.header.frame_id = self.parent_frame
        odom.child_frame_id = self.child_frame
    
        odom.pose.pose.position.x = self.xk[-3]
        odom.pose.pose.position.y = self.xk[-2]

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w
        odom.pose.covariance = covar

        self.odom_pub.publish(odom)
     
        self.tf_broadcaster.sendTransform((self.xk[-3], self.xk[-2], 0.0), q , rospy.Time.now(), self.child_frame, self.parent_frame)
   
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

        path_marker.scale.x = 0.02  # Line width
        path_marker.color = ColorRGBA(0.0, 1.0, 0.0, 1)  # Green, opaque

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

        path_marker.scale.x = 0.02  # Line width
        path_marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)  # Red, semi-transparent

        for (x, y) in self.gt_path:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            path_marker.points.append(point)

        self.gt_trajectory_pub.publish(path_marker)  
    
    def publish_viewpoints(self):
        '''This function publishes the viewpoints as markers'''

        # Create a marker array message
        marker_frontier_lines = MarkerArray()
        marker_frontier_lines.markers = []

        viewpoints_list = []

        for i in range(0,len(self.xk),3):
            myMarker = Marker()
            myMarker.header.frame_id = "world_ned"
            myMarker.type = myMarker.SPHERE
            myMarker.action = myMarker.ADD
            myMarker.id = i

            myMarker.pose.orientation.x = 0.0
            myMarker.pose.orientation.y = 0.0
            myMarker.pose.orientation.z = 0.0
            myMarker.pose.orientation.w = 1.0

            myPoint = Point()
            myPoint.x = self.xk[i]
            myPoint.y = self.xk[i+1]

            myMarker.pose.position = myPoint
            myMarker.color=ColorRGBA(0.224, 1, 0, 0.35)

            myMarker.scale.x = 0.1
            myMarker.scale.y = 0.1
            myMarker.scale.z = 0.05
            viewpoints_list.append(myMarker)

        self.viewpoints_pub.publish(viewpoints_list)

    
    # Define function to create marker message for eigenvalues and eigenvectors
    def publish_covariance_marker(self):
        '''This function publishes the covariance eigenvalues and eigenvectors as markers'''
        # Create marker array message
        marker_array  = MarkerArray()
        # Define marker properties
        id = 0
        markers = []
        for j in range(0,len(self.xk),3):
            
            eigenvalues, eigenvectors = eig(self.Pk[j:j+2, j:j+2])
            marker = Marker()
            marker.header.frame_id = "world_ned"  # Set the frame 
            
            marker.type   =  Marker.SPHERE
            marker.action =  Marker.ADD
            marker_scale = [0.1, 0.1, 0.1]
            if np.any(np.isnan(eigenvalues)) or np.any(np.isinf(eigenvalues)) or np.any(eigenvalues < 0):
                # print("Invalid eigenvalues:", eigenvalues)
                marker_scale  = [0.1, 0.1, 0.00001]
            else:
                marker_scale  = [2.4*np.sqrt(eigenvalues[0]), 2.4*np.sqrt(eigenvalues[1]) , 0.0001 ]

            marker_color  = [1.0, 1.0, 0.0, 1.0]  # Red colo
            id += 1
            marker.id = id 
            marker.scale.x = marker_scale[0]
            marker.scale.y = marker_scale[1]
            marker.scale.z = 0.1
            marker.color.r = marker_color[0]
            marker.color.g = marker_color[1]
            marker.color.b = marker_color[2]
            marker.color.a = marker_color[3]
            marker.pose.position.x = self.xk[j]
            marker.pose.position.y = self.xk[j+1]
            marker.pose.position.z = 0.0
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            quat = tf.transformations.quaternion_from_euler(0, 0, angle)
            # quat = tf.transformations.quaternion_from_matrix(matrix)
            # Normalize the quaternion
            quat_magnitude = np.sqrt(quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2)
            quat = [q/quat_magnitude for q in quat]

            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            markers.append(marker)

            
        marker_array.markers.extend(markers)

        self.covariance_markers.publish(marker_array)

    # -------------------------
    # Map creation functions
        #1_create_map
    # -------------------------
    def create_map(self):
        """
        Creates and publishes the current EKF SLAM map as a PointCloud2 message.

        This function:
        - Calls build_map() to get the full set of scan points transformed into world coordinates.
        - Stacks all scans into a single array, adding z=0.0 for each point (2D map in 3D space).
        - Publishes the map to /slam/map for RViz visualization.

        Notes:
            - Frame ID is set to self.parent_frame (e.g., 'world_ned').
            - Points are published as (x, y, z=0.0) for compatibility with 3D PointCloud2.
        """
        z = np.zeros((len(self.map), 1))
        full_map = np.zeros((0,3))
        map = build_map(self.scan,self.xk)
        for m in map:
            if(len(m) == 0):
                continue
            z = np.zeros((len(m), 1))
            full_map = np.block([[full_map] ,[m,z]])
        # Create the header for the point cloud message
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world_ned'  # Set the frame ID
        # # Create the point cloud message
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, full_map)
        self.full_map_pub.publish(point_cloud_msg)

# -------------------------
# Main Entry Point
# -------------------------
# if __name__ == '__main__': 
#     rospy.init_node('differential_robot_node') # Initialize the node
#     diff_drive = DifferentialDrive()           # Create an instance of the DifferentialDrive class
#     rospy.spin() 

if __name__ == '__main__': 
    rospy.init_node('differential_robot_node') # Initialize the node
    diff_drive = DifferentialDrive()           # Create instance

    rospy.on_shutdown(diff_drive.save_logs_to_csv)  # Save logs on shutdown
    rospy.spin()


