"""启动 MuJoCo 仿真桥，可选一并起导航。

默认 `use_nav:=true`，会同时启动 `tests/nav_node.py`，
这样 Foxglove 一连上就能看到 /costmap、/plan，不需要再手动开第二个终端。

只调仿真、不想要导航时：

    ros2 launch mujoco_ros2_bridge bridge.launch.py use_nav:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
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
    use_nav = LaunchConfiguration('use_nav')
    workspace = LaunchConfiguration('workspace')

    return LaunchDescription([
        DeclareLaunchArgument('scene', default_value='rc2027/scene_nav.xml',
                              description='场景 XML（相对本包 model/ 或绝对路径）'),
        DeclareLaunchArgument('use_viewer', default_value='true',
                              description='是否打开 MuJoCo viewer'),
        DeclareLaunchArgument('use_foxglove', default_value='true',
                              description='是否启动 foxglove_bridge'),
        DeclareLaunchArgument('use_nav', default_value='true',
                              description='是否同时启动导航节点（发布 /costmap 和 /plan）'),
        DeclareLaunchArgument(
            'workspace',
            default_value=os.path.join(os.path.expanduser('~'), 'ros2_ws'),
            description='工作区根目录（含 tests/nav_node.py）'),

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

        # 导航节点：发布 /costmap、/plan，订阅 /goal_pose，输出 /cmd_vel
        # 用 ExecuteProcess 而不是 Node，因为它不在 ROS 包里（只是脚本）
        ExecuteProcess(
            cmd=['python3', 'tests/nav_node.py'],
            cwd=workspace,
            output='screen',
            condition=IfCondition(use_nav),
        ),
    ])
