cmake_minimum_required(VERSION 3.22)
project(pekf_slam_icp LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)

include(cmake/learn_state_estimation.cmake)  # target: state_estimation

# ---- ROS-free algorithm core
add_library(pekf_slam_core STATIC src/pose_ekf_slam.cpp)
target_include_directories(pekf_slam_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(pekf_slam_core PUBLIC state_estimation Eigen3::Eigen)

# ---- ROS 2 nodes
add_executable(pekf_slam_node src/pekf_slam_node.cpp)
target_link_libraries(pekf_slam_node pekf_slam_core)
ament_target_dependencies(pekf_slam_node rclcpp geometry_msgs nav_msgs sensor_msgs visualization_msgs tf2 tf2_ros)

add_executable(lap_driver_node src/lap_driver_node.cpp)
target_include_directories(lap_driver_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(lap_driver_node rclcpp geometry_msgs nav_msgs sensor_msgs std_msgs)

install(TARGETS pekf_slam_node lap_driver_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/evaluate_run.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(test_pose_ekf_slam test/test_pose_ekf_slam.cpp TIMEOUT 300)
    target_link_libraries(test_pose_ekf_slam pekf_slam_core)
    ament_add_gtest(test_lap_follower test/test_lap_follower.cpp)
    target_include_directories(test_lap_follower PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

ament_package()
