from setuptools import setup
import os
from glob import glob

package_name = 'racecar_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=['ILQR', 'utils'],
    package_dir={'': 'scripts'},  # Your source files are in scripts/
    data_files=[
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),      # launch files
        ('share/' + package_name + '/configs', glob('configs/*.yaml')),         # config files
        ('share/' + package_name, ['package.xml']),                             # package manifest
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='iLQR-based planner for autonomous racing',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'traj_planning_node = scripts.traj_planning_node:main',  # entry point to run your node
        ],
    },
)
