#!/usr/bin/python3
"""Republish the RPLidar LaserScan as a PointCloud2 on ``/cloud_in`` (input for octomap_server)."""

import sensor_msgs.point_cloud2 as pc2
import rospy
from sensor_msgs.msg import PointCloud2, LaserScan
import laser_geometry.laser_geometry as lg


rospy.init_node("laserscan_to_pointcloud")

lp = lg.LaserProjection()

pc_pub = rospy.Publisher("/cloud_in", PointCloud2, queue_size=1)


def scan_cb(msg):
    """Project the incoming LaserScan to PointCloud2 and publish it."""
    pc2_msg = lp.projectLaser(msg)
    pc_pub.publish(pc2_msg)
    rospy.loginfo_once("Published PointCloud2 from LaserScan.") #comment/uncomment++++

    
#rospy.Subscriber("/scan", LaserScan, scan_cb, queue_size=1)  # not publishing anything
rospy.Subscriber("/turtlebot/kobuki/sensors/rplidar", LaserScan, scan_cb, queue_size=1)
rospy.spin()