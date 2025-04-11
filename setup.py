from setuptools import setup
import os
from glob import glob

package_name = 'racecar_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=['ILQR', 'utils'],
    package_dir={'': 'scripts'},
    data_files=[
        # Install package.xml
        ('share/' + package_name, ['package.xml']),
        
        # Install launch files (both .launch and .launch.py)
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch*')),
        
        # Install config YAMLs and model files
        (os.path.join('share', package_name, 'configs'), glob('configs/*.yaml') + glob('configs/*.sav')),
        
        # Register resource index
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='YOUR_NAME',
    maintainer_email='YOUR_EMAIL@domain.com',
    description='iLQR-based planner for autonomous racing cars with dynamic/static obstacle avoidance and PWM control.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ROS 2 launchable script entry
            'traj_planning_node = scripts.traj_planning_node:main',
        ],
    },
)
