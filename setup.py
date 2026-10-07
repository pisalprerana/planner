from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'output'),
            glob('planner/output/*.csv'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bot',
    maintainer_email='pisalprerana@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'waypoint_array_dubins = planner.waypoint_array_dubins:main',
            'dubins_reference_path_publisher = planner.dubins_reference_path_publisher:main',
            'gps_imu_tf_broadcaster = planner.gps_imu_tf_broadcaster:main',
            'path_frame_transformer = planner.path_frame_transformer:main',
            'coverage_path_live_viewer = planner.coverage_path_live_viewer:main',
            'trajectory_logger = planner.trajectory_logger:main',

        ],
    },
)
