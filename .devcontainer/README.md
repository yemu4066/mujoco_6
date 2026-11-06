# 开发容器（可选）

**只有本地开发需要这个目录。交付给主逻辑时不需要。**

`nav/` 是纯 Python 库，只要有 numpy/scipy/casadi 就能跑，不需要容器。
这里配的是「跑仿真 + 看 MuJoCo 窗口」的环境。

## 用法

```bash
cd mujoco_6
code .
# 左下角绿色按钮 → Reopen in Container
```

或者命令行：

```bash
docker compose -f .devcontainer/docker-compose.yml up -d
docker exec -it mujoco_6 bash
```

## ⚠️ 容器名冲突

配置里 `container_name: mujoco_6`。如果**已经有一个叫 `mujoco_6` 的容器**（手动起的），
`docker compose up` 会报：

```
Error response from daemon: Conflict. The container name "/mujoco_6" is already in use
```

先删掉旧的：

```bash
docker rm -f mujoco_6
```

**注意**：删了之后再 up 是**全新容器**，里面装的 Mesa 库会丢。
不过 `postCreateCommand` 会自动重装，等 1~2 分钟就行。

## 同一时间只能跑一个本镜像的容器

两个容器都会起 `rmw_zenohd`，抢 7447 端口。切换前先停另一个：

```bash
docker stop mujoco_5      # 或 mujoco_6
```

## 环境变量

`docker-compose.yml` 里设了：

```yaml
RMW_IMPLEMENTATION=rmw_zenoh_cpp
ROS_LOCALHOST_ONLY=1
```

**这两个必须一致**，否则 `ros2 topic list` 看不到对方的节点。

如果手动 `docker exec` 进去发现 `echo $RMW_IMPLEMENTATION` 是空的：

```bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ROS_LOCALHOST_ONLY=1
```

## 首次进入做什么

```bash
cd /home/Elaina/ros2_ws
python3 -c "import nav; print('ok')"
```

如果要跑仿真，需要先编译 ROS 包：

```bash
mkdir -p src
ln -sfn ../../sim/nav_interfaces src/nav_interfaces
ln -sfn ../../sim/mujoco_ros2_bridge src/mujoco_ros2_bridge
colcon build --symlink-install --packages-select nav_interfaces mujoco_ros2_bridge
```

（当前分支如果没有 `sim/`，跳过这步）
