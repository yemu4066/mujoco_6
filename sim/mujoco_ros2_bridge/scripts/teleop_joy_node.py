#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class TeleopJoyNode(Node):
    def __init__(self):
        super().__init__('teleop_joy_node')
        
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10)
        
        # 修正标准映射：
        # Axis 1：左摇杆上下 (前后移动 -> Twist.linear.x)
        # Axis 0：左摇杆左右 (横向侧移 -> Twist.linear.y)
        # Axis 3：右摇杆左右 (原地旋转 -> Twist.angular.z)
        self.declare_parameter('axis_linear_x', 1)  
        self.declare_parameter('axis_linear_y', 0)  
        self.declare_parameter('axis_angular_z', 3) 
        self.declare_parameter('scale_linear', 2.0)
        self.declare_parameter('scale_angular', 2.0)

    def joy_callback(self, joy_msg):
        self.get_logger().info(f'Joy axes: {joy_msg.axes}')
        twist = Twist()
        
        axis_x = self.get_parameter('axis_linear_x').value
        axis_y = self.get_parameter('axis_linear_y').value
        axis_z = self.get_parameter('axis_angular_z').value
        scale_l = self.get_parameter('scale_linear').value
        scale_a = self.get_parameter('scale_angular').value

        DEADZONE = 0.05

        # 1. 前后控制 (X轴)
        # 向上推是负值，所以加负号实现前进
        if abs(joy_msg.axes[axis_x]) > DEADZONE:
            twist.linear.x = joy_msg.axes[axis_x] * scale_l
        
        # 2. 左右侧移 (Y轴)
        # 向左推是正值，所以直接乘 scale 实现向左平移
        if abs(joy_msg.axes[axis_y]) > DEADZONE:
            twist.linear.y = -joy_msg.axes[axis_y] * scale_l
        
        # 3. 旋转控制
        if len(joy_msg.axes) > axis_z:
            if abs(joy_msg.axes[axis_z]) > DEADZONE:
                twist.angular.z = joy_msg.axes[axis_z] * scale_a
        
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = TeleopJoyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
