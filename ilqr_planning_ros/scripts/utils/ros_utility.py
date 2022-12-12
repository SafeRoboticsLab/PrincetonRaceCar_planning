import rospy


def get_ros_param(param_name, default):
    '''
    Read a parameter from the ROS parameter server. If the parameter does not exist, return the default value.
    '''
    if rospy.has_param(param_name):
        return rospy.get_param(param_name)
    else:
        rospy.logwarn("Parameter '%s' not found, using default: %s", param_name, default)
        return default