from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare environment variable equivalent to <env name="LD_PRELOAD" ... />
    preload_env = SetEnvironmentVariable(
        name='LD_PRELOAD',
        value='/usr/lib/aarch64-linux-gnu/libgomp.so.1'
    )

    # Declare all arguments previously used as <arg ... />
    args = [
        DeclareLaunchArgument('output', default_value='screen'),
        DeclareLaunchArgument('odom_topic', default_value='/SLAM/Pose'),
        DeclareLaunchArgument('control_topic', default_value='/Control'),
        DeclareLaunchArgument('obstacle_topic', default_value='/Prediction/Obstacles'),
        DeclareLaunchArgument('traj_topic', default_value='/Planning/Trajectory'),
        DeclareLaunchArgument('path_topic', default_value='/Routing/Path'),
        DeclareLaunchArgument('receding_horizon', default_value='false'),
        DeclareLaunchArgument('replan_dt', default_value='0.1'),
        DeclareLaunchArgument('ilqr_config', default_value='configs/lab1.yaml'),
    ]

    # Create the Trajectory Planner node
    planner_node = Node(
        package='racecar_planner',
        executable='traj_planning_node',  # maps to the setup.py entry point
        name='traj_planning',
        output=LaunchConfiguration('output'),
        parameters=[{
            'odom_topic': LaunchConfiguration('odom_topic'),
            'control_topic': LaunchConfiguration('control_topic'),
            'obstacle_topic': LaunchConfiguration('obstacle_topic'),
            'traj_topic': LaunchConfiguration('traj_topic'),
            'path_topic': LaunchConfiguration('path_topic'),
            'package_path': FindPackageShare('racecar_planner'),
            'simulation': False,
            'replan_dt': LaunchConfiguration('replan_dt'),
            'receding_horizon': LaunchConfiguration('receding_horizon'),
            'ilqr_params_file': PathJoinSubstitution([
                FindPackageShare('racecar_planner'),
                LaunchConfiguration('ilqr_config')
            ]),
            'PWM_model': PathJoinSubstitution([
                FindPackageShare('racecar_planner'),
                'configs/mlp_model.sav'
            ]),
        }]
    )

    return LaunchDescription(args + [preload_env, planner_node])
