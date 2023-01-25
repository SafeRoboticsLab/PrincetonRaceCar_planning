#!/usr/bin/env python

import threading
import rospy
import csv
import time
import numpy as np
import os

from .utils import RealtimeBuffer, get_ros_param, State2D, Policy
from .iLQR import iLQR, RefPath

from racecar_msgs.msg import ServoMsg #, OdomMsg, PathMsg, ObstacleMsg, PathMsg
# https://wiki.ros.org/rospy_tutorials/Tutorials/numpy
# from rospy.numpy_msg import numpy_msg
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg # used to display the trajectory on RVIZ
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
        self.t_last_state = None
        
        # Indicate if the planner is used to generate a new trajectory
        self.planner_ready = False
        
        # Indicate if the planner is first time to run rather than replan
        self.planner_stopped = True
        
        self.setup_publisher()
        
        self.setup_subscriber()

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
        self.odom_topic = get_ros_param('~odom_topic', '/slam_pose')
        self.path_topic = get_ros_param('~path_topic', '/Routing/Path')
        self.obstacle_topic = get_ros_param('~obstacle_topic', '/prediction/obstacles')
        
        # Read ROS topic names to publish
        self.control_topic = get_ros_param('~control_topic', '/control/servo_control')
        self.traj_topic = get_ros_param('~traj_topic', '/Planning/Trajectory')
        # self.if_pub_control = get_ros_param('~publish_control', False)
        
        # Read Planning parameters
        # if true, the planner will load a path from a file rather than subscribing to a path topic
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
        if self.ilqr_params_file == "":
            ilqr_params_abs_path = None
        elif os.path.isabs(self.ilqr_params_file):
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
        self.trajectory_pub = rospy.Publisher(self.traj_topic, PathMsg, queue_size=1)

        # Publisher for the control command
        self.control_pub = rospy.Publisher(self.control_topic, ServoMsg, queue_size=1)
            
    def setup_subscriber(self):
        '''
        This function sets up the subscriber for the odometry and path
        '''
        self.pose_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odometry_callback, queue_size=1)

        self.path_sub = rospy.Subscriber(self.path_topic, PathMsg, self.path_callback, queue_size=1)

    def setup_service(self):
        '''
        Set up ros service
        '''
        self.start_srv = rospy.Service('/planning/start_planning', Empty, self.start_planning_cb)
        self.stop_srv = rospy.Service('/planning/stop_planning', Empty, self.stop_planning_cb)

    def start_planning_cb(self, req):
        '''ros service callback function for start planning'''
        print(req)
        rospy.loginfo("Start planning!")
        self.planner_ready = True
        return EmptyResponse()

    def stop_planning_cb(self, req):
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
        state_cur = State2D(odom_msg = odom_msg)

        # We first get the control command from the buffer
        self.publish_control(state_cur)
        
        # Add the current state to the buffer
        # Planning thread will read from the buffer
        self.state_buffer.writeFromNonRT(state_cur)

    def obstacle_callback(self, obstacle_msg):
        pass
    
    def path_callback(self, path_msg):
        x = []
        y = []
        width_L = []
        width_R = []
        speed_limit = []
        
        for waypoint in path_msg.poses:
            x.append(waypoint.pose.position.x)
            y.append(waypoint.pose.position.y)
            width_L.append(waypoint.pose.orientation.x)
            width_R.append(waypoint.pose.orientation.y)
            speed_limit.append(waypoint.pose.orientation.z)
                    
        centerline = np.array([x, y])
        ref_path = RefPath(centerline, width_L, width_R, speed_limit, loop=False)
        self.path_buffer.writeFromNonRT(ref_path)
    
    def publish_control(self, state: State2D):
        
        servo_msg = ServoMsg()
        servo_msg.header.stamp = rospy.Time.now()
            
        if self.t_last_state is None:
            dt_state = 0.0
        else:
            dt_state = state.t - self.t_last_state
        
        policy = self.policy_buffer.readFromRT()
        if policy is not None:
            x_i, u_i, K_i = policy.get_policy(state.t)
            cur_state_vec = state.state_vector(self.prev_delta)
            dx = (cur_state_vec - x_i)
            dx[3] = np.mod(dx[3] + np.pi, 2 * np.pi) - np.pi
            u = u_i + K_i @ dx
            
            servo_msg.throttle = u[0]
            servo_msg.steer = self.prev_delta + u[1]*dt_state
            self.prev_delta = x_i[-1]
        else:
            # apply brake
            servo_msg.throttle = -5.0
            servo_msg.steer = 0.0
        self.control_pub.publish(servo_msg)
        self.t_last_state = state.t

    def planning_thread(self):
        # time.sleep(5)
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
                        self.planner.update_ref_path(new_path)
                        rospy.logdebug("Path updated!")
                    
                    # Replan use ilqr
                    state_vec = state_cur.state_vector(self.prev_delta)
                    new_plan = self.planner.plan(state_vec, init_controls, verbose=False)
                    
                    plan_status = new_plan['status']
                    if plan_status == -1:
                        rospy.logwarn_once("No path specified!")
                        continue
                    
                    if self.planner_stopped:
                        # Since the planner was previously stopped, we need to reset the time
                        self.planner_stopped = False
                        t_cur = rospy.get_time()
                        rospy.loginfo("First plan gerenated!")
                    
                    if self.planner_ready:
                        # If stop planning is called, we will not write to the buffer
                        new_policy = Policy(new_plan['states'], new_plan['controls'],
                                            new_plan['K_closed_loop'], t_cur, 
                                            self.planner.dt, self.planner.n)
                        self.policy_buffer.writeFromNonRT(new_policy)

                        self.t_last_replan = t_cur
                        # publish the new policy for RVIZ visualization
                        self.trajectory_pub.publish(new_policy.to_msg())        

    def run(self):
        rospy.spin() 