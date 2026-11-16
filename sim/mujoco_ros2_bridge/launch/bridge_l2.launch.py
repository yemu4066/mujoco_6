"""L2 对抗专用场景一键启动（6×6 平面布局，无高度差）。

场地布局（俯视，6×6 m，全部压平到地面层）：

      y=+3  ┌─────────────────────────┐
            │                         │
      y= 1.5│      ┌───────────┐      │
            │      │           │      │
      y= 0.9│──────┤    L2     ├──────│   ← 东西走廊（唯一可通行）
            │      │   3×3     │      │
      y=-0.9│──────┤           ├──────│
            │      └───────────┘      │
      y=-1.5│                         │
            │                         │
      y=-3  └─────────────────────────┘
           x=-3   x=-1.5    x=1.5   x=3

"申"字形 = 中央 3×3 的 L2 方框 + 东西两条贯穿走廊。
南北半场被墙封死 —— 对应"双方只能通过各自侧的台阶进入 L2"，
实车从西/东走廊进入，红蓝各占一侧。

车起点 (-2.6, 0)，朝东，正好在西走廊入口。

没有动态障碍：mujoco_node 找不到 obstacle_l1/l2 会打印 warning 后跳过。
中央有规则里的固定基座（Ø270 × 高 800）作为障碍。

用法：
    ros2 launch mujoco_ros2_bridge bridge_l2.launch.py

可选覆盖：
    map:=maps/l2_arena.yaml          换地图（默认就是这个）
    scene:=rc2027/field.xml          换回真实场地
    use_viewer:=false                无头模式
    use_foxglove:=false              不起 foxglove
    workspace:=/path/to/mujoco_6     指定 nav_node.py 所在根目录

注意：本文件假设 ros2_ws 里 src/mujoco_ros2_bridge 是指向
      mujoco_6/sim/mujoco_ros2_bridge 的符号链接（或在 mujoco_6 里
      直接 colcon build），这样 model/l2/scene_l2.xml 能被
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
            'scene', default_value='l2/scene_l2.xml',
            description='场景 XML（相对本包 model/ 或绝对路径）'),
        DeclareLaunchArgument(
            'map', default_value='maps/l2_arena.yaml',
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
