#!/usr/bin/env python
import rospy
import tf
import math
from sensor_msgs.msg import JointState, Imu, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped

class DeadReckoning:
    def __init__(self):
        rospy.init_node('dead_reckoning_node')

        # Robot parameters (adjust bias here to simulate calibration errors)
        self.wheel_radius = 0.035 * 1.05  # 5% radius bias
        self.wheel_base = 0.23 * 0.95     # 5% shorter base bias

        # Pose and velocity
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = rospy.Time.now()

        # Wheel positions
        self.last_left_pos = None
        self.last_right_pos = None
        self.left_pos = None
        self.right_pos = None

        self.use_imu = True

        # Publishers and subscribers
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()
        rospy.Subscriber('/turtlebot/joint_states', JointState, self.joint_callback)
        rospy.Subscriber('/turtlebot/kobuki/sensors/imu_data', Imu, self.imu_callback)
        rospy.Subscriber('/turtlebot/kobuki/sensors/rplidar', LaserScan, self.lidar_callback)

    def joint_callback(self, msg):
        if 'turtlebot/kobuki/wheel_left_joint' in msg.name:
            idx = msg.name.index('turtlebot/kobuki/wheel_left_joint')
            self.left_pos = msg.position[idx]
        elif 'turtlebot/kobuki/wheel_right_joint' in msg.name:
            idx = msg.name.index('turtlebot/kobuki/wheel_right_joint')
            self.right_pos = msg.position[idx]

        if self.left_pos is None or self.right_pos is None:
            return

        if self.last_left_pos is None or self.last_right_pos is None:
            self.last_left_pos = self.left_pos
            self.last_right_pos = self.right_pos
            return

        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        if dt == 0:
            return

        d_left = self.wheel_radius * (self.left_pos - self.last_left_pos)
        d_right = self.wheel_radius * (self.right_pos - self.last_right_pos)
        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / self.wheel_base

        self.x += d_center * math.cos(self.theta + d_theta / 2.0)
        self.y += d_center * math.sin(self.theta + d_theta / 2.0)
        if not self.use_imu:
            self.theta += d_theta

        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = 'world_ned'
        odom.child_frame_id = 'turtlebot/kobuki/base_footprint'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        q = tf.transformations.quaternion_from_euler(0, 0, self.theta)
        odom.pose.pose.orientation = Quaternion(*q)

        odom.twist.twist.linear.x = d_center / dt
        odom.twist.twist.angular.z = d_theta / dt

        self.odom_pub.publish(odom)

        self.tf_broadcaster.sendTransform(
            (self.x, self.y, 0.0),
            q,
            current_time,
            'turtlebot/kobuki/base_footprint',
            'world_ned'
        )

        self.last_left_pos = self.left_pos
        self.last_right_pos = self.right_pos
        self.last_time = current_time

    def imu_callback(self, msg):
        if not self.use_imu:
            return
        orientation_q = msg.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
        self.theta = yaw

    def lidar_callback(self, scan):
        # Just forwarding to RViz. In RViz, add LaserScan display with topic:
        # /turtlebot/kobuki/sensors/rplidar
        pass

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    dr = DeadReckoning()
    dr.run()
