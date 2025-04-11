#!/usr/bin/env python3

# Standard Python libraries
import os
import sys

# ROS 2 Python client library
import rclpy
from rclpy.node import Node

# ✅ Import your main node class from the ROS 2 port of traj_planner.py
from traj_planner import TrajectoryPlannerNode


def main(args=None):
    # ---------------------------------------------------------------
    # [Optional] Limit GPU memory usage for JAX/XLA (used by ILQR)
    # This prevents JAX from pre-allocating the entire GPU.
    # Value is a fraction (e.g., "0.2" = 20% of total GPU memory).
    # ---------------------------------------------------------------
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"

    # ---------------------------------------------------------------
    # Initialize the ROS 2 client library
    # This must be done before creating any nodes.
    # ---------------------------------------------------------------
    rclpy.init(args=args)

    # Log a startup message to the ROS 2 console (rclpy uses Python logging)
    rclpy.logging.get_logger("traj_planning_node").info("Starting trajectory planning node...")

    # ---------------------------------------------------------------
    # Create the planner node (defined in traj_planner.py)
    # This node sets up all publishers, subscribers, services, and threads
    # ---------------------------------------------------------------
    planner_node = TrajectoryPlannerNode()

    try:
        # ---------------------------------------------------------------
        # Spin the node — this blocks until shutdown and keeps ROS callbacks running
        # Any subscriptions, timers, or services defined in the node will now function
        # ---------------------------------------------------------------
        rclpy.spin(planner_node)

    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C shutdown from terminal
        rclpy.logging.get_logger("traj_planning_node").info("Shutting down planner node due to keyboard interrupt.")

    finally:
        # ---------------------------------------------------------------
        # Cleanly destroy the node and shutdown rclpy
        # Always do this at the end of your main to avoid resource leaks
        # ---------------------------------------------------------------
        planner_node.destroy_node()
        rclpy.shutdown()


# ---------------------------------------------------------------
# Standard Python entry point check
# Ensures this script only runs when called directly (not when imported)
# ---------------------------------------------------------------
if __name__ == '__main__':
    main()
