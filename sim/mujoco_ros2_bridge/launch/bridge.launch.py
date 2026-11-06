"""单独启动 MuJoCo 仿真桥（不含导航）。

通常直接用 nav_bringup 的 bringup.launch.py；
这个文件用于只调仿真、不想起导航的场合。
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory('mujoco_ros2_bridge')
    params = os.path.join(share, 'config', 'params.yaml')

    scene = LaunchConfiguration('scene')
    use_viewer = LaunchConfiguration('use_viewer')
    use_foxglove = LaunchConfiguration('use_foxglove')

    return LaunchDescription([
        DeclareLaunchArgument('scene', default_value='rc2027/scene_nav.xml',
                              description='场景 XML（相对本包 model/ 或绝对路径）'),
        DeclareLaunchArgument('use_viewer', default_value='true',
                              description='是否打开 MuJoCo viewer'),
        DeclareLaunchArgument('use_foxglove', default_value='true',
                              description='是否启动 foxglove_bridge'),

        Node(
            package='mujoco_ros2_bridge',
            executable='mujoco_node.py',
            name='mujoco_node',
            output='screen',
            parameters=[
                params,
                {
                    'scene': ParameterValue(scene, value_type=str),
                    'use_viewer': ParameterValue(use_viewer, value_type=bool),
                },
            ],
        ),

        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
            condition=IfCondition(use_foxglove),
            # 重启整套系统时 8765 可能还在 TIME_WAIT，绑定失败会自动重试
            respawn=True,
            respawn_delay=3.0,
        ),
    ])
