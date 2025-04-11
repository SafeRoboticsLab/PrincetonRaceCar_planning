import numpy as np
from nav_msgs.msg import Path as TrajMsg
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler
from builtin_interfaces.msg import Time

class Policy:
    """
    A container class to store and evaluate a feedback policy.
    """
    def __init__(self, X, U, K, t0, dt, T) -> None:
        self.nominal_X = X
        self.nominal_U = U
        self.K = K
        self.t0 = t0
        self.dt = dt
        self.T = T

    def get_policy(self, t):
        i = self.get_index(t)
        if i >= (self.T - 1):
            return None, None, None
        return self.nominal_X[:, i], self.nominal_U[:, i], self.K[:, :, i]

    def get_policy_by_state(self, x):
        distance = np.linalg.norm(self.nominal_X[:2, :].T - x[:2], axis=1)
        i = np.argmin(distance)
        if distance[i] > 0.5:
            return None, None, None
        return self.nominal_X[:, i], self.nominal_U[:, i], self.K[:, :, i]

    def get_index(self, t):
        return int(np.ceil((t - self.t0) / self.dt))

    def get_ref_controls(self, t):
        i = self.get_index(t)
        if i >= self.T:
            return None
        ref_u = np.zeros_like(self.nominal_U)
        ref_u[:, :self.T - i] = self.nominal_U[:, i:]
        return ref_u

    def to_msg(self, node_clock, frame_id='map'):
        """
        Generate a nav_msgs/Path message from the nominal trajectory.

        Args:
            node_clock: node.get_clock() from rclpy
            frame_id (str): frame to attach to each pose

        Returns:
            TrajMsg: ROS2 Path message
        """
        traj_msg = TrajMsg()
        traj_msg.header.frame_id = frame_id
        traj_msg.header.stamp = node_clock.now().to_msg()

        for i in range(self.T):
            t = self.t0 + i * self.dt
            pose = self.state_to_pose_stamped(self.nominal_X[:, i], t, frame_id)
            traj_msg.poses.append(pose)

        return traj_msg

    @staticmethod
    def state_to_pose_stamped(state, t, frame_id='map'):
        """
        Convert a [x, y, v, yaw, delta] state vector to a PoseStamped.

        Args:
            state: (5,) state vector
            t: time in seconds (float)
            frame_id: frame for the pose

        Returns:
            PoseStamped
        """
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp.sec = int(t)
        pose.header.stamp.nanosec = int((t % 1.0) * 1e9)
        pose.pose.position.x = state[0]
        pose.pose.position.y = state[1]
        pose.pose.position.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, state[3])
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def __str__(self):
        return f"Policy: t0: {self.t0}, dt: {self.dt}, N: {self.T}\n" + \
               f"nominal_X: {self.nominal_X}\n" + \
               f"nominal_U: {self.nominal_U}\n"
