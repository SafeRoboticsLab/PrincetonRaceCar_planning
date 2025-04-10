cmake_minimum_required(VERSION 3.5)
project(racecar_planner)

## Find ament macros and libraries
## ROS2 replaces catkin with ament
find_package(ament_python REQUIRED)

# ----------------------------------------------------------------------------
# Python installation
# ----------------------------------------------------------------------------

# This macro ensures modules and global scripts declared in setup.py get installed
# See: https://docs.ros.org/en/foxy/How-To-Guides/Build-a-Python-Package.html
ament_python_install_package(${PROJECT_NAME})

# ----------------------------------------------------------------------------
# Declare ROS messages, services and actions (not used here)
# ----------------------------------------------------------------------------
## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "rosidl_default_generators"
##   * add a exec_depend tag for each package in MSG_DEP_SET
## * In this file (CMakeLists.txt):
##   * add "rosidl_default_generators" to find_package(...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the rosidl_generate_interfaces entry below
##   * add every package in MSG_DEP_SET to DEPENDENCIES below

# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

# rosidl_generate_interfaces(${PROJECT_NAME}
#   ${msg_files}
#   DEPENDENCIES std_msgs
# )

# ----------------------------------------------------------------------------
# Declare ROS dynamic reconfigure parameters (not used in ROS2)
# ----------------------------------------------------------------------------
## In ROS1, dynamic_reconfigure was used to define tunable parameters with .cfg files.
## In ROS2, this is replaced by parameter declaration APIs inside each Node.
## So this section is removed in ROS2:
##
## generate_dynamic_reconfigure_options(
##   cfg/planner.cfg
## )

# ----------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------

# Mark executable scripts (Python etc.) for installation
# ROS2 Python scripts are installed to lib/${PROJECT_NAME}
install(
  PROGRAMS
    scripts/traj_planning_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ----------------------------------------------------------------------------
# Ament-specific configuration
# ----------------------------------------------------------------------------

# ament_package is similar to catkin_package in ROS1
# It finalizes the build and generates export files
ament_package()
