import rclpy
from rclpy.node import Node
from typing import Any

def get_ros_param(node: Node, param_name: str, default: Any) -> Any:
    """
    Retrieve a parameter from a ROS2 Node. If the parameter is not declared yet,
    declare it with a default value and return the default.

    Args:
        node (rclpy.node.Node): The node to access parameters from.
        param_name (str): The name of the parameter to retrieve.
        default (Any): The default value to use if parameter is undeclared.

    Returns:
        Any: The parameter value.
    """
    if not node.has_parameter(param_name):
        # Declare the parameter with a default if it hasn't been declared yet
        node.declare_parameter(param_name, default)
        node.get_logger().info(
            f"Parameter '{param_name}' not declared. Declaring with default value: {default}"
        )

    value = node.get_parameter(param_name).value
    node.get_logger().info(f"Parameter '{param_name}' loaded with value: {value}")
    return value
