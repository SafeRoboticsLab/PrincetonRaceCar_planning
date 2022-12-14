#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from planner import PlanningRecedingHorizon
import sys, os


def main():
    rospy.init_node('traj_planning_node')
    rospy.loginfo("Start trajectory planning node")


    planner = PlanningRecedingHorizon()
    planner.run()


if __name__ == '__main__':
    main()
