cmake_minimum_required(VERSION 3.5)
project(racecar_planner)

# ----------------------------------------------------------------------------
# Required for Python-based ROS 2 packages
# ----------------------------------------------------------------------------
find_package(ament_python REQUIRED)

# ----------------------------------------------------------------------------
# Install Python packages defined in setup.py
# ----------------------------------------------------------------------------
ament_python_install_package(${PROJECT_NAME})

# ----------------------------------------------------------------------------
# Install executable scripts (Python nodes)
# These are entry-point files you want to be executable directly
# ----------------------------------------------------------------------------
install(
  PROGRAMS
    scripts/traj_planning_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ----------------------------------------------------------------------------
# Install shared files (launch, config, model)
# ----------------------------------------------------------------------------
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  DIRECTORY configs/
  DESTINATION share/${PROJECT_NAME}/configs
)

# Install model file if it's not inside configs/
# (You can remove this block if mlp_model.sav is already inside configs/)
# install(
#   FILES configs/mlp_model.sav
#   DESTINATION share/${PROJECT_NAME}/configs
# )

# ----------------------------------------------------------------------------
# Finalize with ament configuration
# ----------------------------------------------------------------------------
ament_package()
