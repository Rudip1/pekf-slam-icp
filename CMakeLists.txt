cmake_minimum_required(VERSION 3.0.2)
project(localization)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  src/slam_icp_node.py
  src/test_node.py
  src/utils_script/laser_scan_to_point_cloud_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# The nodes import utils_script from the directory they live in.
install(DIRECTORY src/utils_script
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  PATTERN "__pycache__" EXCLUDE
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
