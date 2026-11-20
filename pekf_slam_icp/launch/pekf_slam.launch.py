"""Simulation + pose-based EKF SLAM, optionally with the scripted driver, the evaluation and RViz.

    ros2 launch pekf_slam_icp pekf_slam.launch.py                         # drive yourself (teleop on /cmd_vel)
    ros2 launch pekf_slam_icp pekf_slam.launch.py drive:=true evaluate:=true gui:=false rviz:=false
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory("pekf_slam_icp"))
    sim = Path(get_package_share_directory("pekf_slam_icp_gazebo")) / "launch" / "sim.launch.py"
    params = str(share / "config" / "pekf_slam.yaml")
    arg = LaunchConfiguration

    evaluator = Node(package="pekf_slam_icp", executable="evaluate_run.py", output="screen",
                     parameters=[{"use_sim_time": True, "output": arg("output")}],
                     condition=IfCondition(arg("evaluate")))
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true", description="Gazebo window"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("drive", default_value="false", description="scripted laps through the office"),
        DeclareLaunchArgument("laps", default_value="2"),
        DeclareLaunchArgument("use_imu", default_value="true", description="IMU heading updates"),
        DeclareLaunchArgument("evaluate", default_value="false",
                              description="compare with ground truth when the driver finishes, then shut down"),
        DeclareLaunchArgument("output", default_value="pekf_run.csv", description="CSV written by the evaluation"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(sim)),
                                 launch_arguments={"gui": arg("gui")}.items()),
        Node(package="pekf_slam_icp", executable="pekf_slam_node", name="pekf_slam", output="screen",
             parameters=[params, {"use_imu": arg("use_imu")}]),
        Node(package="pekf_slam_icp", executable="lap_driver_node", output="screen",
             parameters=[{"use_sim_time": True, "laps": arg("laps")}], condition=IfCondition(arg("drive"))),
        evaluator,
        RegisterEventHandler(OnProcessExit(target_action=evaluator, on_exit=[EmitEvent(event=Shutdown())])),
        Node(package="rviz2", executable="rviz2", arguments=["-d", str(share / "rviz" / "pekf_slam.rviz")],
             parameters=[{"use_sim_time": True}], condition=IfCondition(arg("rviz"))),
    ])
