#!/bin/bash
# 一键启动仿真 + 导航（在容器内跑）
#
# 用法：
#   bash tests/start_all.sh        # 启动
#   bash tests/start_all.sh stop   # 停止
#   bash tests/start_all.sh status # 看状态

export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}
export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-1}
source /opt/conda/setup.bash
cd /home/Elaina/ros2_ws || exit 1
source install/setup.bash

PATTERN='[r]os2 launch mujoco|[m]ujoco_ros2_bridge/lib|[f]oxglove_bridge|[n]av_node.py'

alive() { ps -eo stat,cmd | grep -vE '^Z' | grep -cE "$PATTERN"; }

# 统计时排除 bash -c 包装壳（docker exec -d 会留下一层）
count() { ps -eo stat,cmd | grep -vE '^Z' | grep -E "$1" | grep -vc '^S.*bash -c'; }

status() {
    echo "  launch    : $(count '[r]os2 launch mujoco')"
    echo "  mujoco    : $(count '[m]ujoco_ros2_bridge/lib')"
    echo "  foxglove  : $(count '[f]oxglove_bridge')"
    echo "  nav_node  : $(count '[n]av_node.py')"
}

stop_all() {
    echo "停止中..."
    for r in 1 2 3; do
        for p in $(ps -eo pid,cmd | grep -E "$PATTERN" | grep -v grep | awk '{print $1}'); do
            kill -9 "$p" 2>/dev/null
        done
        sleep 2
        [ "$(alive)" = "0" ] && break
    done
    [ "$(alive)" = "0" ] && echo "  ✓ 已全部停止" || echo "  ✗ 还有 $(alive) 个"
}

start_all() {
    echo "启动仿真..."
    setsid nohup ros2 launch mujoco_ros2_bridge bridge.launch.py \
        > /tmp/launch.log 2>&1 < /dev/null &
    for i in $(seq 1 40); do
        sleep 1
        grep -q "MuJoCo viewer 已启动" /tmp/launch.log 2>/dev/null && { echo "  ✓ 仿真就绪"; break; }
    done

    echo "启动导航..."
    setsid nohup python3 tests/nav_node.py > /tmp/navnode.log 2>&1 < /dev/null &
    for i in $(seq 1 40); do
        sleep 1
        grep -q "就绪：发" /tmp/navnode.log 2>/dev/null && { echo "  ✓ 导航就绪"; break; }
    done

    echo
    status
    echo
    echo "Foxglove: ws://127.0.0.1:8765"
}

case "${1:-start}" in
    start)  stop_all > /dev/null 2>&1; start_all ;;
    stop)   stop_all ;;
    status) status ;;
    *) echo "用法: $0 [start|stop|status]"; exit 1 ;;
esac
