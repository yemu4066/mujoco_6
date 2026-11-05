# 单个圆形动态障碍（世界坐标系，单位 m / m/s）。
# 这是导航栈与"障碍来源"之间的唯一契约：
#   仿真真值、感知检测、或者手工发布，都走同一种消息。
std_msgs/Header header

uint32 id
geometry_msgs/Point position      # 障碍中心
geometry_msgs/Vector3 velocity    # 世界系速度（可选，缺省为 0）
float32 radius                    # 外接圆半径
float32 height                    # 高度（0 表示未知，仅用于可视化/分层）
float32 confidence                # 置信度 0~1（1 表示真值）
