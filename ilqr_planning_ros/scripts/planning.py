#!/usr/bin/env python

import threading
import rospy
import csv
import yaml
import time
import numpy as np
# https://wiki.ros.org/rospy_tutorials/Tutorials/numpy
from rospy.numpy_msg import numpy_msg

from .iLQR import Policy, Path

from .utils import RealtimeBuffer
from princeton_racecar_msgs.msg import ServoMsg, OdomMsg, PathMsg, ObstacleMsg, TrajMsg


        
class Planning_MPC():

    def __init__(self):
        '''
        Main class for the MPC trajectory planner
        '''

        self.read_parameters()
        # set up the optimal control solver
        self.setup_planner()
        
        # create buffers to handle multi-threading
        self.state_buffer = RealtimeBuffer()
        self.policy_buffer = RealtimeBuffer()
        self.path_buffer = RealtimeBuffer()
        self.obstacle_buffer = RealtimeBuffer()
        
        self.t_last_replan = 0
        
        self.setup_publisher()
        
        self.setup_subscriber()

        # start planning thread
        threading.Thread(target=self.planning_thread).start()

    def read_parameters(self):
        '''
        This function reads the parameters from the parameter server
        '''
        
        # Read ROS topic names to subscribe 
        self.odometry_topic = rospy.get_param('~odometry_topic', '/perception/odometry')
        self.path_topic = rospy.get_param('~path_topic', '/planning/path')
        self.obstacle_topic = rospy.get_param('~obstacle_topic', '/prediction/obstacles')
        
        # Read ROS topic names to publish
        self.control_topic = rospy.get_param('~control_topic', '/controller/rc_control')
        self.trajectory_topic = rospy.get_param('~trajectory_topic', '/planning/trajectory')
        self.if_pub_control = rospy.get_param('~publish_control', False)
        
        # Read Planning parameters
        self.predefined_path = rospy.get_param('~predefined_path', False)
        # if true, the planner will load a path from a file rather than subscribing to a path topic
        self.path_file = rospy.get_param('~path_file', None)
        self.path_width_left = rospy.get_param('~path_width_L', 0.5)
        self.path_width_right = rospy.get_param('~path_width_R', 0.5)
        self.path_loop = rospy.get_param('~path_loop', False)
            
        self.replan_dt = rospy.get_param('~replan_dt', 0.1)
        
        self.ilqr_params_file = rospy.get_param('~ilqr_params_file', 'configs/ilqr.yaml')
        
    def setup_planner(self):
        '''
        This function setup the iLQR solver
        '''
        # Read iLQR parameters
        with open(self.ilqr_params_file) as file:
            ilqr_params = yaml.load(file, Loader=yaml.FullLoader)
            
        # TODO: Initialize iLQR solver
        self.planner = None

    def setup_publisher(self):
        '''
        This function sets up the publisher for the trajectory
        '''
        self.trajectory_pub = rospy.Publisher(self.trajectory_topic, TrajMsg, queue_size=1)
        if self.if_pub_control:
            self.control_pub = rospy.Publisher(self.control_topic, ServoMsg, queue_size=1)
            
    def setup_subscriber(self):
        '''
        This function sets up the subscriber for the odometry and path
        '''
        self.pose_sub = rospy.Subscriber(self.odometry_topic, OdomMsg, self.odometry_callback, queue_size=1)
        self.obstacle_sub = rospy.Subscriber(self.obstacle_topic, ObstacleMsg, self.obstacle_callback, queue_size=1)
        if self.predefined_path:
            path_center_line = self.load_path(self.path_file)
            path = Path(path_center_line, self.path_width_left, self.path_width_right, self.path_loop)
            self.path_buffer.writeFromNonRT(path)
        else:
            self.path_sub = rospy.Subscriber(self.path_topic, numpy_msg(PathMsg), self.path_callback, queue_size=1)

    def odometry_callback(self, odom_msg):
        """
        Subscriber callback function of the robot pose
        """
        # TODO: write the new pose to the buffer
        cur_policy = self.policy_buffer.readFromRT()
        if self.if_pub_control and cur_policy is not None:
            pass
            # TODO: publish the control command given the feedback policy
        
        # write the new pose to the buffer
        self.state_buffer.writeFromNonRT(odom_msg)

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
    
    
    def publish_control(self, v, u, cur_t):
        control = ServoMsg()
        control.header.stamp = cur_t
        a = u[0]
        delta = -u[1]
        
        if a<0:
            d = a/10-0.5
        else:
            temp = np.array([v**3, v**2, v, a**3, a**2, a, v**2*a, v*a**2, v*a, 1])
            d = temp@self.d_open_loop
            d = d+min(delta*delta*0.5,0.05)
        
        control.throttle = np.clip(d, -1.0, 1.0)
        control.steer = np.clip(delta/0.3, -1.0, 1.0)
        control.reverse = False
        self.control_pub.publish(control)

    def planning_thread(self):
        time.sleep(5)
        rospy.loginfo("Planning thread started")
        while not rospy.is_shutdown():
            # determine if we need to replan
            replan = False
            if self.state_buffer.new_data_available:
                state_cur = self.state_buffer.readFromRT()
            
                t_cur = state_cur.header.stamp.to_sec()

                if (t_cur - self.t_last_replan) >= self.replan_dt:
                    replan = True
            
                # if we have a updated path, replan immediately
                if self.path_buffer.new_data_available:
                    new_path = self.path_buffer.readFromRT()                    
                    # TODO: re-initialize the planner with new path
                    replan = True
                    
                if replan:
                    # TODO: do a replan
                    new_policy = None
                    self.policy_buffer.writeFromNonRT(new_policy)
                    self.t_last_replan = t_cur         
                    
                    # TODO: publish the new trajectory   

    def run(self):
        rospy.spin() 