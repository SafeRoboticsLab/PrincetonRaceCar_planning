import numpy as np
import pickle
from .ros_utility import get_ros_param
from rclpy.node import Node

class GeneratePwm:
    """
    Convert acceleration and steering angle into PWM signals
    using a learned open-loop model (e.g., MLP) for ESC/servo control.
    """
    def __init__(self, node: Node):
        """
        Constructor: loads parameters and model.
        Args:
            node (Node): rclpy Node instance for param/log access
        """
        self.node = node
        self.read_parameters()
        self.mlp_model = pickle.load(open(self.model_path, 'rb'))

    def read_parameters(self):
        """
        Load safety and model configuration parameters.
        """
        self.max_throttle = get_ros_param(self.node, 'max_throttle', 0.5)
        self.min_throttle = get_ros_param(self.node, 'min_throttle', -0.3)
        self.model_path = get_ros_param(self.node, 'PWM_model', 'model.pkl')

    def convert(self, accel: float, steer: float, v: float):
        """
        Convert acceleration & steering to PWM commands.

        Args:
            accel (float): Linear acceleration [m/s²]
            steer (float): Steering angle [rad]
            v (float): Current velocity [m/s]

        Returns:
            Tuple[float, float]: (throttle_pwm, steer_pwm)
        """
        # Limit speed for safety
        if v > 3:
            accel = min(accel, 0)
            v_bounded = 3
        else:
            v_bounded = v

        # Convert steering angle to PWM (bounded)
        steer_pwm = -np.clip(steer / 0.37, -1, 1)
        accel_bounded = np.sign(accel) * min(abs(accel), 2 + v)

        # Prepare model input
        input_vec = np.array([[accel_bounded, v_bounded, np.abs(steer_pwm)]])

        # Check for NaN
        if np.any(np.isnan(input_vec)):
            self.node.get_logger().warn("Control input contains NaN!")
            return self.min_throttle, steer_pwm

        # Predict throttle PWM
        d = self.mlp_model.predict(input_vec)[0]
        throttle_pwm = np.clip(d, self.min_throttle, self.max_throttle)

        # Add steering compensation at low speeds
        if v < 0.2:
            throttle_pwm += np.abs(steer_pwm) * 0.04

        return throttle_pwm, steer_pwm
