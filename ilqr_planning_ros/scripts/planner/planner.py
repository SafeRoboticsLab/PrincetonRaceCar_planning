#!/usr/bin/env python

import threading
import rospy
import csv
import time
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

from .utils import RealtimeBuffer, get_ros_param
from .policy import Policy
from .iLQR import iLQR

# from racecar_msgs.msg import ServoMsg, OdomMsg, PathMsg, ObstacleMsg, TrajMsg
# https://wiki.ros.org/rospy_tutorials/Tutorials/numpy
from rospy.numpy_msg import numpy_msg
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as TrajMsg # used to display the trajectory on RVIZ

from std_srvs.srv import Empty, EmptyResponse

class State2D():
    '''
    2D vehicle state
    '''
    def __init__(self) -> None:
        self.x = 0 #x position
        self.y = 0 #y position
        self.theta = 0 # pose angle around z axis
        
        self.w = 0 # angular velocity around z axis
        self.v = 0
        self.v_dir = 0 # direction of the velocity
        self.v_long = 0 # longitudinal velocity
        self.v_lat = 0 # lateral velocity
        
        self.initialized = False
        
    def from_SE3(self, pose_base_to_world, twist_base, decimal=3):
        '''
        Construct state from SE3 pose and twist
            pose_base_to_world: SE3 pose from world to base
            twist_base: twist of the base
        '''
        self.x = np.round(pose_base_to_world[0,3], decimal)
        self.y = np.round(pose_base_to_world[1,3], decimal)
        
        rot = R.from_matrix(pose_base_to_world[:3,:3])
        self.theta = rot.as_euler('zyx', degrees=False)[0]
        
        self.v_long = np.round(twist_base[0,3], decimal)
        self.v_lat = np.round(twist_base[1,3], decimal)
        
        self.v = np.sqrt(self.v_long*self.v_long + self.v_lat*self.v_lat)
        self.v_dir = np.arctan2(self.v_lat, self.v_long)
        
        self.w = np.round(twist_base[2,0], decimal) # angular velocity around z axis
        
        self.initialized = True
    
    def from_state(self, x, y, theta, v_long, w):
        '''
        Construct state from a state object
        '''
        self.x = x
        self.y = y
        self.theta = theta
        
        self.w = w
        
        # assume no slip
        self.v = v_long
        self.v_dir = 0 
        self.v_long = v_long
        self.v_lat = 0
        
        self.initialized = True
        
    def __str__(self):
        return f"State at [{np.round(self.x,3)}, {np.round(self.y,3)}] "+ \
            f"with pose {np.round(np.rad2deg(self.theta),3)} deg\n" + \
            f"Speed: {np.round(self.v,3)} pointing to {np.round(np.rad2deg(self.v_dir),3)} deg; Omega: {np.round(self.w,3)} \n"+ \
            f"v_long: {self.v_long}; v_lat: {self.v_lat}; \n"

class PlanningRecedingHorizon():
    '''
    Main class for the Receding Horizon trajectory planner
    '''

    def __init__(self):

        self.read_parameters()
        # set up the optimal control solver
        self.setup_planner()

        self.t_last_replan = 0
        
        self.run_planner = False
        
        self.setup_subscriber()

        self.setup_publisher()

        self.setup_service()

        # # start planning thread
        threading.Thread(target=self.planning_thread).start()

    def read_parameters(self):
        '''
        This function reads the parameters from the parameter server
        '''
        # Required parameters
        self.package_path = rospy.get_param('~package_path')
        # Read transformation parameters
        '''  Cam
            T
             Base
        '''
        self.T_axis_to_cam = rospy.get_param('~T_axis_to_cam')
        
        self.T_axis_to_cam = np.array(self.T_axis_to_cam).reshape(4,4)
        
        # Read ROS topic names to subscribe 
        self.odometry_topic = get_ros_param('~odometry_topic', '/perception/odometry')
        self.path_topic = get_ros_param('~path_topic', '/planning/path')
        self.obstacle_topic = get_ros_param('~obstacle_topic', '/prediction/obstacles')
        
        # Read ROS topic names to publish
        self.control_topic = get_ros_param('~control_topic', '/controller/rc_control')
        self.trajectory_topic = get_ros_param('~trajectory_topic', '/planning/trajectory')
        # self.if_pub_control = get_ros_param('~publish_control', False)
        
        # Read Planning parameters
        self.predefined_path = get_ros_param('~predefined_path', False)
        # if true, the planner will load a path from a file rather than subscribing to a path topic
        self.path_file = get_ros_param('~path_file', None)
        self.path_width_left = get_ros_param('~path_width_L', 0.5)
        self.path_width_right = get_ros_param('~path_width_R', 0.5)
        self.path_loop = get_ros_param('~path_loop', False)
            
        self.replan_dt = get_ros_param('~replan_dt', 0.1)
        
        self.ilqr_params_file = get_ros_param('~ilqr_params_file', 'configs/ilqr.yaml')
        
    def setup_planner(self):
        '''
        This function setup the iLQR solver
        '''
        # Read iLQR parameters
        if os.path.isabs(self.ilqr_params_file):
            ilqr_params_abs_path = self.ilqr_params_file
        else:
            ilqr_params_abs_path = os.path.join(self.package_path, self.ilqr_params_file)
        
        # TODO: Initialize iLQR solver
        self.planner = iLQR(ilqr_params_abs_path)

        # create buffers to handle multi-threading
        self.state_buffer = RealtimeBuffer()
        self.policy_buffer = RealtimeBuffer()
        self.path_buffer = RealtimeBuffer()
        self.obstacle_buffer = RealtimeBuffer()

    def setup_publisher(self):
        '''
        This function sets up the publisher for the trajectory
        '''
        # Publisher for the planned nominal trajectory for visualization
        self.trajectory_pub = rospy.Publisher(self.trajectory_topic, TrajMsg, queue_size=1)

        # Publisher for the control command
        self.control_pub = rospy.Publisher(self.control_topic, ServoMsg, queue_size=1)
            
    def setup_subscriber(self):
        '''
        This function sets up the subscriber for the odometry and path
        '''
        self.pose_sub = rospy.Subscriber(self.odometry_topic, Odometry, self.odometry_callback, queue_size=1)
        self.obstacle_sub = rospy.Subscriber(self.obstacle_topic, ObstacleMsg, self.obstacle_callback, queue_size=1)
        if self.predefined_path:
            path_center_line = self.load_path(self.path_file)
            path = Path(path_center_line, self.path_width_left, self.path_width_right, self.path_loop)
            self.path_buffer.writeFromNonRT(path)
        else:
            self.path_sub = rospy.Subscriber(self.path_topic, numpy_msg(PathMsg), self.path_callback, queue_size=1)

    def setup_service(self):
        '''
        Set up ros service
        '''
        self.start_srv = rospy.Service('start_planning', Empty, self.start_planning)
        self.stop_srv = rospy.Service('stop_planning', Empty, self.stop_planning)

    def start_planning(self):
        '''ros service callback function for start planning'''
        rospy.loginfo("Start planning!")
        self.run_planner = True
        return EmptyResponse()

    def stop_planning(self):
        '''ros service callback function for stop planning'''
        rospy.loginfo("Stop planning!")
        self.run_planner = False
        self.policy_buffer.reset()
        return EmptyResponse()

    def odometry_callback(self, odom_msg):
        """
        Subscriber callback function of the robot pose
        """
        
        # Retrive pose and twist from the odometry message
        T_cam_to_world = np.eye(4)
        T_cam_to_world[:3,:3] = R.from_quat([odom_msg.pose.pose.orientation.x,
                                        odom_msg.pose.pose.orientation.y,
                                        odom_msg.pose.pose.orientation.z,
                                        odom_msg.pose.pose.orientation.w]).as_matrix()
        
        T_cam_to_world[:3,3] = np.array([odom_msg.pose.pose.position.x,
                                        odom_msg.pose.pose.position.y,
                                        odom_msg.pose.pose.position.z])
        
        T_axis_to_world = np.dot(T_cam_to_world, self.T_axis_to_cam)
        
        twist_cam = np.zeros([4,4])
        twist_cam[:3, :3] = np.array([[0, -odom_msg.twist.twist.angular.z, odom_msg.twist.twist.linear.y],
                                    [odom_msg.twist.twist.angular.z, 0, -odom_msg.twist.twist.linear.x],
                                    [-odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.x, 0]])
        
        twist_cam[:3, 3] = np.array([odom_msg.twist.twist.linear.x,
                                    odom_msg.twist.twist.linear.y,
                                    odom_msg.twist.twist.linear.z])
        
        T_cam_to_axis = np.linalg.inv(self.T_axis_to_cam)
        
        '''
             base    base   cam    cam
            V     = T    * V    * T
                     cam           base
        '''
        twist_axis = np.dot(T_cam_to_axis, np.dot(twist_cam, self.T_axis_to_cam))
        
        state_axis = State2D()
        state_axis.from_SE3(T_axis_to_world, twist_axis)
        print(state_axis)
        # # TODO: write the new pose to the buffer
        # cur_policy = self.policy_buffer.readFromRT()
        # if self.if_pub_control and cur_policy is not None:
        #     pass
        #     # TODO: publish the control command given the feedback policy
        
        # # write the new pose to the buffer
        # self.state_buffer.writeFromNonRT(odom_msg)

    def obstacle_callback(self, obstacle_msg):
        pass
    
    def path_callback(self, path_msg):
        pass
    
    def load_path(self, filepath: str):
        """
        Gets the centerline of the track from the trajectory data. We currently only
        support 2D track.

        Args:
            filepath (str): the path to file consisting of the centerline position.

        Returns:
            np.ndarray: centerline, of the shape (2, N).
        """
        x = []
        y = []
        with open(filepath) as f:
            spamreader = csv.reader(f, delimiter=',')
            for i, row in enumerate(spamreader):
                if i > 0:
                    x.append(float(row[0]))
                    y.append(float(row[1]))

        return np.array([x, y])
    
    
    # def publish_control(self, v, u, cur_t):
    #     control = ServoMsg()
    #     control.header.stamp = cur_t
    #     a = u[0]
    #     delta = -u[1]
        
    #     if a<0:
    #         d = a/10-0.5
    #     else:
    #         temp = np.array([v**3, v**2, v, a**3, a**2, a, v**2*a, v*a**2, v*a, 1])
    #         d = temp@self.d_open_loop
    #         d = d+min(delta*delta*0.5,0.05)
        
    #     control.throttle = np.clip(d, -1.0, 1.0)
    #     control.steer = np.clip(delta/0.3, -1.0, 1.0)
    #     control.reverse = False
    #     self.control_pub.publish(control)

    def planning_thread(self):
        time.sleep(5)
        rospy.loginfo("Planning thread started waiting for ROS service calls...")
        while not rospy.is_shutdown():
            # determine if we need to replan
            if self.state_buffer.new_data_available:
                state_cur = self.state_buffer.readFromRT()
            
                t_cur = state_cur.header.stamp.to_sec()
                dt = t_cur - self.t_last_replan

                # Do replanning
                if dt >= self.replan_dt and self.run_planner:
                    # Get the initial controls for hot start
                    init_controls = None

                    original_policy = self.policy_buffer.readFromRT()
                    if original_policy is not None:
                        init_controls = original_policy.get_ref_controls(t_cur)

                    # Update the obstacles
                    if self.obstacle_buffer.new_data_available:
                        obstacles = self.obstacle_buffer.readFromRT()
                        self.planner.update_obstacles(obstacles)

                    # Update the path
                    if self.path_buffer.new_data_available:
                        new_path = self.path_buffer.readFromRT()
                        self.planner.update_path(new_path)
                        # new path, we ignore the initial controls
                        init_controls = None
                    
                    # Replan use ilqr
                    new_plan = self.planner.plan(state_cur, init_controls)
                    
                    new_policy = Policy(new_plan['states'], new_plan['controls'],
                                        new_plan['K_closed_loop'], t_cur, 
                                        self.planner.dt, self.planner.n)

                    self.policy_buffer.writeFromNonRT(new_policy)
                    
                    self.t_last_replan = t_cur         

    def run(self):
        rospy.spin() 