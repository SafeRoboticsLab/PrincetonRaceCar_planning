#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from traj_planner import TrajectoryPlanner
import sys, os




if __name__ == '__main__':
    # Safe guard for GPU memory
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "20" 
    rospy.init_node('traj_planning_node')
    rospy.loginfo("Start trajectory planning node")

    planner = TrajectoryPlanner()
    rospy.spin()
