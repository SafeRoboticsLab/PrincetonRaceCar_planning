from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare arguments
    launch_args = [
        DeclareLaunchArgument('output', default_value='screen'),
        DeclareLaunchArgument('odom_topic', default_value='/SLAM/Pose'),
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
        DeclareLaunchArgument('ilqr_config', default_value='configs/lab2_task2.yaml'),
        DeclareLaunchArgument('num_dyn_obs', default_value='2'),
        DeclareLaunchArgument('num_static_obs', default_value='0'),
        DeclareLaunchArgument('static_obs_size', default_value='0.18'),
        DeclareLaunchArgument('static_obs_topic', default_value='/Obstacles/Static'),
        DeclareLaunchArgument('dyn_obs_topic', default_value='/Obstacles/Dynamic'),
    ]

    # Get paths
    racecar_planner_dir = get_package_share_directory('racecar_planner')
    racecar_interface_dir = get_package_share_directory('racecar_interface')
    racecar_obs_detection_dir = get_package_share_directory('racecar_obs_detection')
    racecar_routing_dir = get_package_share_directory('racecar_routing')

    # Nodes
    traffic_sim_node = Node(
        package='racecar_interface',
        executable='traffic_simulation_node.py',
        name='traffic_simulation_node',
        output=LaunchConfiguration('output'),
        parameters=[{
            'map_file': os.path.join(racecar_routing_dir, 'cfg', 'track.pkl'),
            'num_dyn_obs': LaunchConfiguration('num_dyn_obs'),
            'num_static_obs': LaunchConfiguration('num_static_obs'),
            'static_obs_size': LaunchConfiguration('static_obs_size'),
            'static_obs_topic': LaunchConfiguration('static_obs_topic'),
            'dyn_obs_topic': LaunchConfiguration('dyn_obs_topic'),
            'pub_rate': LaunchConfiguration('pub_rate'),
        }]
    )

    frs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(racecar_obs_detection_dir, 'launch', 'frs.launch.py')
        ),
        launch_arguments={
            'dyn_obs_topic': LaunchConfiguration('dyn_obs_topic')
        }.items()
    )

    traj_planner_node = Node(
        package='racecar_planner',
        executable='traj_planning_node.py',
        name='traj_planning',
        output=LaunchConfiguration('output'),
        parameters=[{
            'odom_topic': LaunchConfiguration('odom_topic'),
            'control_topic': LaunchConfiguration('control_topic'),
            'obstacle_topic': LaunchConfiguration('obstacle_topic'),
            'traj_topic': LaunchConfiguration('traj_topic'),
            'path_topic': LaunchConfiguration('path_topic'),
            'package_path': racecar_planner_dir,
            'simulation': False,
            'receding_horizon': LaunchConfiguration('receding_horizon'),
            'replan_dt': LaunchConfiguration('replan_dt'),
            'ilqr_params_file': LaunchConfiguration('ilqr_config'),
            'PWM_model': os.path.join(racecar_planner_dir, 'configs', 'mlp_model.sav'),
        }]
    )

    return LaunchDescription(launch_args + [
        traffic_sim_node,
        frs_launch,
        traj_planner_node
    ])
