#!/usr/bin/env python

import threading
import rospy
import csv
import time
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

from .utils import RealtimeBuffer, get_ros_param, State2D, Policy, policy_to_path
from .iLQR import iLQR, Path

# from racecar_msgs.msg import ServoMsg, OdomMsg, PathMsg, ObstacleMsg, TrajMsg
# https://wiki.ros.org/rospy_tutorials/Tutorials/numpy
from rospy.numpy_msg import numpy_msg
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as TrajMsg # used to display the trajectory on RVIZ

from std_srvs.srv import Empty, EmptyResponse


class PlanningRecedingHorizon():
    '''
    Main class for the Receding Horizon trajectory planner
    '''

    def __init__(self):

        self.read_parameters()
        # set up the optimal control solver
        self.setup_planner()

        self.t_last_replan = 0
        
        # Indicate if the planner is used to generate a new trajectory
        self.planner_ready = False
        
        # Indicate if the planner is first time to run rather than replan
        self.planner_stopped = True
        
        self.setup_subscriber()

        self.setup_publisher()

        self.setup_service()

        # start planning thread
        threading.Thread(target=self.planning_thread).start()

    def read_parameters(self):
        '''
        This function reads the parameters from the parameter server
        '''
        # Required parameters
        self.package_path = rospy.get_param('~package_path')
        
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

        self.prev_delta = 0.0

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
        # self.obstacle_sub = rospy.Subscriber(self.obstacle_topic, ObstacleMsg, self.obstacle_callback, queue_size=1)
        if self.predefined_path:
            path_center_line = self.load_path(self.path_file)
            path = Path(path_center_line, self.path_width_left, self.path_width_right, self.path_loop)
            self.path_buffer.writeFromNonRT(path)
        else:
            # self.path_sub = rospy.Subscriber(self.path_topic, numpy_msg(PathMsg), self.path_callback, queue_size=1)
            pass

    def setup_service(self):
        '''
        Set up ros service
        '''
        self.start_srv = rospy.Service('start_planning_cb', Empty, self.start_planning_cb)
        self.stop_srv = rospy.Service('stop_planning_cb', Empty, self.stop_planning_cb)

    def start_planning_cb(self):
        '''ros service callback function for start planning'''
        rospy.loginfo("Start planning!")
        self.planner_ready = True
        return EmptyResponse()

    def stop_planning_cb(self):
        '''ros service callback function for stop planning'''
        rospy.loginfo("Stop planning!")
        self.planner_ready = False
        self.planner_stopped = True
        self.policy_buffer.reset()
        self.prev_delta = 0.0
        return EmptyResponse()
    
    def odometry_callback(self, odom_msg):
        """
        Subscriber callback function of the robot pose
        """

        # Retrive state needed for planning from the odometry message
        # [x, y, v, w, delta]
        state_cur = State2D()
        state_cur.from_odom_msg(odom_msg)

        # We first get the control command from the buffer
        self.publish_control(state_cur)
        
        # Add the current state to the buffer
        # Planning thread will read from the buffer
        self.state_buffer.writeFromNonRT(state_cur)

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
    
    def publish_control(self, state: State2D):
        pass

    def planning_thread(self):
        time.sleep(5)
        rospy.loginfo("Planning thread started waiting for ROS service calls...")
        while not rospy.is_shutdown():
            # determine if we need to replan
            if self.state_buffer.new_data_available:
                state_cur = self.state_buffer.readFromRT()
            
                t_cur = state_cur.t
                dt = t_cur - self.t_last_replan

                # Do replanning
                if dt >= self.replan_dt and self.planner_ready:
                    # Get the initial controls for hot start
                    init_controls = None

                    original_policy = self.policy_buffer.readFromRT()
                    if original_policy is not None:
                        init_controls = original_policy.get_ref_controls(t_cur)

                    # Update the obstacles
                    if self.obstacle_buffer.new_data_available:
                        obstacles = self.obstacle_buffer.readFromRT()
                        self.planner.update_obstacles(obstacles)
                        rospy.logdebug("Obstacles updated!")

                    # Update the path
                    if self.path_buffer.new_data_available:
                        new_path = self.path_buffer.readFromRT()
                        self.planner.update_path(new_path)
                        # new path, we ignore the initial controls
                        init_controls = None
                        rospy.logdebug("Path updated!")
                    
                    # Replan use ilqr
                    state_vec = state_cur.state_vector(self.prev_delta)
                    new_plan = self.planner.plan(state_vec, init_controls, verbose=False)
                    
                    if self.planner_stopped:
                        # Since the planner was previously stopped, we need to reset the time
                        self.planner_stopped = False
                        t_cur = rospy.get_time()
                        rospy.loginfo("First plan gerenated!")
                    
                    new_policy = Policy(new_plan['states'], new_plan['controls'],
                                        new_plan['K_closed_loop'], t_cur, 
                                        self.planner.dt, self.planner.n)

                    self.policy_buffer.writeFromNonRT(new_policy)
                    
                    self.t_last_replan = t_cur
                    # publish the new policy for RVIZ visualization
                    self.trajectory_pub.publish(new_policy.to_msg())        

    def run(self):
        rospy.spin() 