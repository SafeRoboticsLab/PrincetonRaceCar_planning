from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

def generate_launch_description():
    # ---------------------------------------------
    # Declare arguments (equivalent to <arg> in ROS1)
    # These can be set from the command line
    # ---------------------------------------------
    args = [
        DeclareLaunchArgument('odom_topic', default_value='/Simulation/Pose'),
        DeclareLaunchArgument('control_topic', default_value='/Control'),
        DeclareLaunchArgument('obstacle_topic', default_value='/Prediction/Obstacles'),
        DeclareLaunchArgument('traj_topic', default_value='/Planning/Trajectory'),
        DeclareLaunchArgument('path_topic', default_value='/Routing/Path'),
        DeclareLaunchArgument('pub_rate', default_value='20'),
        DeclareLaunchArgument('receding_horizon', default_value='true'),
        DeclareLaunchArgument('init_x', default_value='2'),
        DeclareLaunchArgument('init_y', default_value='0.15'),
        DeclareLaunchArgument('init_yaw', default_value='0'),
        DeclareLaunchArgument('lane_change_cost', default_value='2'),
        DeclareLaunchArgument('replan_dt', default_value='0.1'),
        DeclareLaunchArgument('ilqr_config', default_value='configs/lab1.yaml'),
    ]

    # ---------------------------------------------
    # Include the racecar_interface simulation.launch.py
    # This replaces the <include> tag from ROS1
    # ---------------------------------------------
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('racecar_interface'),
            '/launch/simulation.launch.py'
        ]),
        launch_arguments={
            'odom_topic': LaunchConfiguration('odom_topic'),
            'control_topic': LaunchConfiguration('control_topic'),
            'pub_rate': LaunchConfiguration('pub_rate'),
            'init_x': LaunchConfiguration('init_x'),
            'init_y': LaunchConfiguration('init_y'),
            'init_yaw': LaunchConfiguration('init_yaw'),
            'enable_routing': 'true',
            'lane_change_cost': LaunchConfiguration('lane_change_cost'),
        }.items()
    )

    # ---------------------------------------------
    # Launch the Trajectory Planner node
    # (Make sure traj_planning_node.py has a proper entry point in setup.py)
    # ---------------------------------------------
    planner_node = Node(
        package='racecar_planner',
        executable='traj_planning_node',  # refers to the entry point in setup.py
        name='traj_planning',
        output='screen',
        parameters=[{
            'odom_topic': LaunchConfiguration('odom_topic'),
            'control_topic': LaunchConfiguration('control_topic'),
            'obstacle_topic': LaunchConfiguration('obstacle_topic'),
            'traj_topic': LaunchConfiguration('traj_topic'),
            'path_topic': LaunchConfiguration('path_topic'),
            'package_path': FindPackageShare('racecar_planner'),
            'simulation': True,
            'receding_horizon': LaunchConfiguration('receding_horizon'),
            'replan_dt': LaunchConfiguration('replan_dt'),
            'ilqr_params_file': PathJoinSubstitution([
                FindPackageShare('racecar_planner'),
                LaunchConfiguration('ilqr_config')
            ]),
            'PWM_model': PathJoinSubstitution([
                FindPackageShare('racecar_planner'),
                'configs/mlp_model.sav'
            ])
        }]
    )

    return LaunchDescription(args + [simulation_launch, planner_node])
