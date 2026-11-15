"""空白场 + 动态避障一键启动。

和 bridge.launch.py 的区别：
  · scene 默认 empty/scene_empty.xml（20×20 平地，无任何场地几何）
  · nav_node 的地图默认 maps/empty_20x20.yaml（origin [-10,-10]，res 0.05）
  · nav_node 一定启动（不再需要手动开第二个终端）
  · 场景里放了两个 mocap 障碍，会被 mujoco_node 按正弦/折线轨迹移动，
    nav_node 通过 /obstacles 感知 → 动态避障

用法：
    ros2 launch mujoco_ros2_bridge bridge_empty.launch.py

可选覆盖：
    map:=maps/debug_empty_10m.yaml   换地图
    scene:=rc2027/field.xml          换回真实场地
    use_viewer:=false                无头模式
    use_foxglove:=false              不起 foxglove
    workspace:=/path/to/mujoco_6     指定 nav_node.py 所在根目录

注意：本文件假设 ros2_ws 里 src/mujoco_ros2_bridge 是指向
      mujoco_6/sim/mujoco_ros2_bridge 的符号链接（或在 mujoco_6 里
      直接 colcon build），这样 model/empty/scene_empty.xml 能被
      get_package_share_directory 找到。找不到时用绝对路径传 scene:=。
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
    map_yaml = LaunchConfiguration('map')
    use_viewer = LaunchConfiguration('use_viewer')
    use_foxglove = LaunchConfiguration('use_foxglove')
    workspace = LaunchConfiguration('workspace')

    return LaunchDescription([
        DeclareLaunchArgument(
            'scene', default_value='empty/scene_empty.xml',
            description='场景 XML（相对本包 model/ 或绝对路径）'),
        DeclareLaunchArgument(
            'map', default_value='maps/empty_20x20.yaml',
            description='nav_node 用的栅格地图（相对仓库根或绝对路径）'),
        DeclareLaunchArgument('use_viewer', default_value='true',
                              description='是否打开 MuJoCo viewer'),
        DeclareLaunchArgument('use_foxglove', default_value='true',
                              description='是否启动 foxglove_bridge'),
        DeclareLaunchArgument(
            'workspace',
            default_value=os.path.join(os.path.expanduser('~'), 'ros2_ws'),
            description='含 tests/nav_node.py 的仓库根目录'),

        # ---- 仿真 ----
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
                    # 空白场没有台面，对手方块直接贴地（rc2027 默认是
                    # 0.875/1.175，站在台面上，不覆盖会悬空）。
                    # 场景里只有 obstacle_l1，l2 的参数留着无害（不会生效）。
                    'obstacle_z_l1': 0.25,
                    'obstacle_z_l2': 0.25,
                },
            ],
        ),

        # ---- Foxglove ----
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
            condition=IfCondition(use_foxglove),
            # 重启时 8765 可能还在 TIME_WAIT，绑定失败会自动重试
            respawn=True,
            respawn_delay=3.0,
        ),

        # ---- 导航（带 map 参数）----
        # 用 ExecuteProcess 因为它只是仓库里的脚本，不是 ament 安装的可执行
        ExecuteProcess(
            cmd=['python3', 'tests/nav_node.py',
                 '--ros-args', '-p', ['map:=', map_yaml]],
            cwd=workspace,
            output='screen',
        ),
    ])
