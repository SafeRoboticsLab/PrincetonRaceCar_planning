#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import MarkerArray
from racecar_msgs.msg import ServoMsg
from racecar_obs_detection.srv import GetFRS

import threading
import queue
import os
import numpy as np
import time
from tf_transformations import euler_from_quaternion

from utils import (
    RealtimeBuffer, get_ros_param, Policy, GeneratePwm,
    get_obstacle_vertices, frs_to_obstacle, frs_to_msg
)
from ILQR import RefPath
from ILQR import ILQR_jax as ILQR


class TrajectoryPlannerNode(Node):
    def __init__(self):
        super().__init__('trajectory_planner')
        self.update_lock = threading.Lock()
        self.latency = 0.0
        self.planner_ready = True
        self.declare_parameters()
        self.read_parameters()
        self.pwm_converter = GeneratePwm(self)
        self.setup_planner()
        self.setup_publishers()
        self.setup_subscribers()
        self.setup_services()
        self.get_logger().info("Trajectory Planner Node initialized")

        threading.Thread(target=self.control_thread, daemon=True).start()
        planning_thread = self.receding_horizon_planning_thread if self.receding_horizon else self.policy_planning_thread
        threading.Thread(target=planning_thread, daemon=True).start()

    def declare_parameters(self):
        self.declare_parameter('package_path', '')
        self.declare_parameter('receding_horizon', False)
        self.declare_parameter('odom_topic', '/slam_pose')
        self.declare_parameter('path_topic', '/Routing/Path')
        self.declare_parameter('control_topic', '/control/servo_control')
        self.declare_parameter('traj_topic', '/Planning/Trajectory')
        self.declare_parameter('static_obs_topic', '/Obstacles/Static')
        self.declare_parameter('simulation', True)
        self.declare_parameter('replan_dt', 0.1)
        self.declare_parameter('ilqr_params_file', '')
        self.declare_parameter('latency', 0.0)

    def read_parameters(self):
        self.package_path = self.get_parameter('package_path').get_parameter_value().string_value
        self.receding_horizon = self.get_parameter('receding_horizon').get_parameter_value().bool_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        self.control_topic = self.get_parameter('control_topic').get_parameter_value().string_value
        self.traj_topic = self.get_parameter('traj_topic').get_parameter_value().string_value
        self.static_obs_topic = self.get_parameter('static_obs_topic').get_parameter_value().string_value
        self.simulation = self.get_parameter('simulation').get_parameter_value().bool_value
        self.replan_dt = self.get_parameter('replan_dt').get_parameter_value().double_value
        self.latency = self.get_parameter('latency').get_parameter_value().double_value
        ilqr_params_file = self.get_parameter('ilqr_params_file').get_parameter_value().string_value
        self.ilqr_params_abs_path = ilqr_params_file if os.path.isabs(ilqr_params_file) else os.path.join(self.package_path, ilqr_params_file) if ilqr_params_file else None

    def setup_planner(self):
        self.planner = ILQR(self.ilqr_params_abs_path)
        self.plan_state_buffer = RealtimeBuffer()
        self.control_state_buffer = RealtimeBuffer()
        self.policy_buffer = RealtimeBuffer()
        self.path_buffer = RealtimeBuffer()
        self.static_obstacle_dict = {}

    def setup_publishers(self):
        self.trajectory_pub = self.create_publisher(Path, self.traj_topic, 10)
        self.control_pub = self.create_publisher(ServoMsg, self.control_topic, 10)
        self.frs_pub = self.create_publisher(MarkerArray, '/vis/FRS', 10)

    def setup_subscribers(self):
        self.create_subscription(Odometry, self.odom_topic, self.odometry_callback, 10)
        self.create_subscription(Path, self.path_topic, self.path_callback, 10)
        self.create_subscription(MarkerArray, self.static_obs_topic, self.static_obstacle_callback, 10)

    def setup_services(self):
        self.create_service(Empty, '/planning/start_planning', self.start_planning_cb)
        self.create_service(Empty, '/planning/stop_planning', self.stop_planning_cb)
        self.frs_client = self.create_client(GetFRS, '/obstacles/get_frs')

    def start_planning_cb(self, request, response):
        self.get_logger().info('Start planning!')
        self.planner_ready = True
        return response

    def stop_planning_cb(self, request, response):
        self.get_logger().info('Stop planning!')
        self.planner_ready = False
        self.policy_buffer.reset()
        return response

    def odometry_callback(self, msg):
        self.control_state_buffer.writeFromNonRT(msg)

    def static_obstacle_callback(self, msg):
        if self.simulation:
            self.static_obstacle_dict.clear()
        for obs in msg.markers:
            obs_id, vertices = get_obstacle_vertices(obs)
            self.static_obstacle_dict[obs_id] = vertices

    def path_callback(self, path_msg):
        try:
            x, y, left, right, speed = [], [], [], [], []
            for pose in path_msg.poses:
                x.append(pose.pose.position.x)
                y.append(pose.pose.position.y)
                left.append(pose.pose.orientation.x)
                right.append(pose.pose.orientation.y)
                speed.append(pose.pose.orientation.z)
            ref_path = RefPath(np.array([x, y]), left, right, speed, loop=False)
            self.path_buffer.writeFromNonRT(ref_path)
            self.get_logger().info('Path received!')
        except Exception as e:
            self.get_logger().warn(f'Invalid path received: {e}')

    @staticmethod
    def compute_control(x, x_ref, u_ref, K_closed_loop):
        dx = x - x_ref
        dx[3] = (dx[3] + np.pi) % (2 * np.pi) - np.pi
        u = u_ref + K_closed_loop @ dx
        return u[0], u[1]

    def control_thread(self):
        rate = self.create_rate(40)
        u_queue = queue.Queue()
        prev_state = None
        prev_u = np.zeros(3)

        def dyn_step(x, u, dt):
            dx = np.array([
                x[2] * np.cos(x[3]),
                x[2] * np.sin(x[3]),
                u[0],
                x[2] * np.tan(u[1] * 1.1) / 0.257,
                0
            ])
            x_new = x + dx * dt
            x_new[2] = max(0, x_new[2])
            x_new[3] = (x_new[3] + np.pi) % (2 * np.pi) - np.pi
            x_new[4] = u[1]
            return x_new

        while rclpy.ok():
            now = self.get_clock().now().nanoseconds * 1e-9
            t_act = now if self.simulation else now + self.latency
            if self.control_state_buffer.new_data_available:
                odom_msg = self.control_state_buffer.readFromRT()
                t_slam = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
                u = np.zeros(3); u[-1] = t_slam
                while not u_queue.empty() and u_queue.queue[0][-1] < t_slam:
                    u = u_queue.get()
                q = odom_msg.pose.pose.orientation
                psi = euler_from_quaternion([q.x, q.y, q.z, q.w])[-1]
                state_cur = np.array([
                    odom_msg.pose.pose.position.x,
                    odom_msg.pose.pose.position.y,
                    odom_msg.twist.twist.linear.x,
                    psi,
                    u[1]
                ])
                for i in range(u_queue.qsize()):
                    u_next = u_queue.queue[i]
                    state_cur = dyn_step(state_cur, u, u_next[-1] - u[-1])
                    u = u_next
                state_cur = dyn_step(state_cur, u, t_act - u[-1])
                self.plan_state_buffer.writeFromNonRT(np.append(state_cur, t_act))
            elif prev_state is not None:
                dt = t_act - prev_u[-1]
                state_cur = dyn_step(prev_state, prev_u, dt)

            accel, steer = -5, 0
            policy = self.policy_buffer.readFromRT()
            if policy is not None:
                x_ref, u_ref, K = policy.get_policy(t_act) if self.receding_horizon else policy.get_policy_by_state(state_cur)
                if x_ref is not None:
                    accel, steer_rate = self.compute_control(state_cur, x_ref, u_ref, K)
                    steer = max(-0.37, min(0.37, prev_u[1] + steer_rate * (t_act - prev_u[-1])))
                else:
                    self.get_logger().warn("Policy expired — clearing.")
                    self.policy_buffer.reset()

            if not self.simulation and state_cur is not None:
                throttle_pwm, steer_pwm = self.pwm_converter.convert(accel, steer, state_cur[2])
            else:
                throttle_pwm, steer_pwm = accel, steer

            msg = ServoMsg()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.throttle = throttle_pwm
            msg.steer = steer_pwm
            self.control_pub.publish(msg)
            u_queue.put(np.array([accel, steer, t_act]))
            prev_state, prev_u = state_cur, np.array([accel, steer, t_act])
            rate.sleep()

    def receding_horizon_planning_thread(self):
        self.get_logger().info("Receding Horizon Planning thread started...")
        t_last_replan = 0
        while rclpy.ok():
            if not self.plan_state_buffer.new_data_available:
                continue
            state_cur = self.plan_state_buffer.readFromRT()
            t_cur = state_cur[-1]
            if t_cur - t_last_replan < self.replan_dt:
                continue

            init_controls = None
            if (original_policy := self.policy_buffer.readFromRT()) is not None:
                init_controls = original_policy.get_ref_controls(t_cur)

            if self.path_buffer.new_data_available:
                self.planner.update_ref_path(self.path_buffer.readFromRT())

            obstacles_list = list(self.static_obstacle_dict.values())
            try:
                t_list = list(t_cur + np.arange(self.planner.T) * self.planner.dt)
                req = GetFRS.Request(); req.time = t_list
                future = self.frs_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                frs_result = future.result()
                obstacles_list.extend(frs_to_obstacle(frs_result))
                self.frs_pub.publish(frs_to_msg(frs_result))
            except Exception as e:
                self.get_logger().warn(f"FRS service failed: {e}")

            self.planner.update_obstacles(obstacles_list)
            new_plan = self.planner.plan(state_cur[:-1], init_controls, verbose=False)
            if new_plan["status"] == -1:
                self.get_logger().warn("ILQR planning failed — no path.")
                continue

            if self.planner_ready:
                new_policy = Policy(
                    X=new_plan["trajectory"],
                    U=new_plan["controls"],
                    K=new_plan["K_closed_loop"],
                    t0=t_cur,
                    dt=self.planner.dt,
                    T=self.planner.T
                )
                self.policy_buffer.writeFromNonRT(new_policy)
                self.trajectory_pub.publish(new_policy.to_msg())
                t_last_replan = t_cur

    def policy_planning_thread(self):
        self.get_logger().info("Open Loop Planning thread started...")
        while rclpy.ok():
            if not (self.path_buffer.new_data_available and self.planner_ready):
                continue

            new_path = self.path_buffer.readFromRT()
            self.planner.update_ref_path(new_path)

            if self.policy_buffer.readFromRT() is not None:
                self.policy_buffer.reset()
                time.sleep(2)

            state = self.plan_state_buffer.readFromRT()[:-1]
            prev_progress = -np.inf
            _, _, progress = new_path.get_closest_pts(state[:2])
            progress = progress[0]

            nominal_trajectory, nominal_controls, K_closed_loop = [], [], []
            while (progress - prev_progress) * new_path.length > 1e-3:
                nominal_trajectory.append(state)
                new_plan = self.planner.plan(state, None, verbose=False)
                nominal_controls.append(new_plan['controls'][:, 0])
                K_closed_loop.append(new_plan['K_closed_loop'][:, :, 0])
                state = new_plan['trajectory'][:, 1]
                prev_progress = progress
                _, _, progress = new_path.get_closest_pts(state[:2])
                progress = progress[0]

            X = np.array(nominal_trajectory).T
            U = np.array(nominal_controls).T
            K = np.transpose(np.array(K_closed_loop), (1, 2, 0))
            T = X.shape[1]
            t0 = self.get_clock().now().nanoseconds * 1e-9

            new_policy = Policy(X=X, U=U, K=K, t0=t0, dt=self.planner.dt, T=T)
            self.policy_buffer.writeFromNonRT(new_policy)
            self.trajectory_pub.publish(new_policy.to_msg())
            self.get_logger().info("Finished planning a new policy.")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
