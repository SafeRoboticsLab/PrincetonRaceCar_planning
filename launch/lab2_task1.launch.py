from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments (mirroring ROS 1 <arg> tags)
    launch_args = [
        DeclareLaunchArgument('output', default_value='screen'),
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
        DeclareLaunchArgument('ilqr_config', default_value='configs/lab2_task1.yaml'),
        DeclareLaunchArgument('num_dyn_obs', default_value='0'),
        DeclareLaunchArgument('num_static_obs', default_value='1'),
        DeclareLaunchArgument('static_obs_size', default_value='0.18'),
        DeclareLaunchArgument('static_obs_topic', default_value='/Obstacles/Static'),
        DeclareLaunchArgument('dyn_obs_topic', default_value='/Obstacles/Dynamic'),
    ]

    # Include the simulation launch file (previously <include file=...>)
    traffic_simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('racecar_interface'),
                'launch',
                'traffic_simulation.launch.py'  # must be ported to ROS 2 too!
            ])
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
            'num_dyn_obs': LaunchConfiguration('num_dyn_obs'),
            'num_static_obs': LaunchConfiguration('num_static_obs'),
            'static_obs_size': LaunchConfiguration('static_obs_size'),
            'static_obs_topic': LaunchConfiguration('static_obs_topic'),
            'dyn_obs_topic': LaunchConfiguration('dyn_obs_topic'),
        }.items()
    )

    # Define the main trajectory planner node
    planner_node = Node(
        package='racecar_planner',
        executable='traj_planning_node',  # maps to entry point from setup.py
        name='traj_planning',
        output=LaunchConfiguration('output'),
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
            ]),
        }]
    )

    return LaunchDescription(launch_args + [traffic_simulation_launch, planner_node])
