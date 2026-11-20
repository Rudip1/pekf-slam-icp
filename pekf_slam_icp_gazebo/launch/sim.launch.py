"""Gazebo Harmonic with the office world, the robot, its robot_state_publisher and the ROS <-> Gazebo bridge.

    ros2 launch pekf_slam_icp_gazebo sim.launch.py              # with the Gazebo window
    ros2 launch pekf_slam_icp_gazebo sim.launch.py gui:=false   # headless (server only)
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("pekf_slam_icp_gazebo"))
    gz_launch = Path(get_package_share_directory("ros_gz_sim")) / "launch" / "gz_sim.launch.py"
    world = str(share / "worlds" / "office.sdf")
    x, y, yaw = LaunchConfiguration("x"), LaunchConfiguration("y"), LaunchConfiguration("yaw")
    robot_description = ParameterValue(Command(["xacro ", str(share / "urdf" / "robot.urdf.xacro")]), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true", description="open the Gazebo window"),
        DeclareLaunchArgument("x", default_value="-5.5"),
        DeclareLaunchArgument("y", default_value="-4.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(gz_launch)),
                                 launch_arguments={"gz_args": f"-r {world}"}.items(),
                                 condition=IfCondition(LaunchConfiguration("gui"))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(gz_launch)),
                                 launch_arguments={"gz_args": f"-r -s --headless-rendering {world}"}.items(),
                                 condition=UnlessCondition(LaunchConfiguration("gui"))),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description, "use_sim_time": True}]),
        Node(package="ros_gz_sim", executable="create", output="screen",
             arguments=["-name", "pekf_robot", "-topic", "robot_description",
                        "-x", x, "-y", y, "-z", "0.02", "-Y", yaw]),
        Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
             parameters=[{"config_file": str(share / "config" / "bridge.yaml"), "use_sim_time": True}]),
    ])
