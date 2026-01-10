import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, Quaternion


def yaw_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def quat_to_yaw(q: Quaternion) -> float:
    return 2.0 * math.atan2(q.z, q.w)


@dataclass
class Particle:
    x: float
    y: float
    theta: float


class MCLPrediction(Node):
    def __init__(self):
        super().__init__("mcl_prediction")  # IMPORTANT

        # Parameters
        self.declare_parameter("num_particles", 100)
        self.declare_parameter("x_min", -10.0)
        self.declare_parameter("x_max", 10.0)
        self.declare_parameter("y_min", -10.0)
        self.declare_parameter("y_max", 10.0)
        self.declare_parameter("frame_id", "map")

        self.declare_parameter("trans_noise_std", 0.05)
        self.declare_parameter("rot_noise_std", 0.02)

        self.declare_parameter("odom_topic", "/robot_noisy")

        self.N = int(self.get_parameter("num_particles").value)
        self.x_min = float(self.get_parameter("x_min").value)
        self.x_max = float(self.get_parameter("x_max").value)
        self.y_min = float(self.get_parameter("y_min").value)
        self.y_max = float(self.get_parameter("y_max").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.trans_noise_std = float(self.get_parameter("trans_noise_std").value)
        self.rot_noise_std = float(self.get_parameter("rot_noise_std").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        # State
        self.particles: List[Particle] = self._init_uniform_particles(self.N)
        self.prev_odom_pose: Optional[Tuple[float, float, float]] = None

        # ROS I/O
        self.pub_particles = self.create_publisher(PoseArray, "particles", 10)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)

        self.publish_particles()

        self.get_logger().info(
            f"MCLPrediction started. N={self.N}, odom_topic={self.odom_topic}, "
            f"trans_noise_std={self.trans_noise_std}, rot_noise_std={self.rot_noise_std}"
        )

    def _init_uniform_particles(self, N: int) -> List[Particle]:
        out = []
        for _ in range(N):
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            th = random.uniform(-math.pi, math.pi)
            out.append(Particle(x=x, y=y, theta=th))
        return out

    def on_odom(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = yaw_to_pi(quat_to_yaw(msg.pose.pose.orientation))

        if self.prev_odom_pose is None:
            self.prev_odom_pose = (x, y, yaw)
            return

        px, py, pyaw = self.prev_odom_pose

        dx = x - px
        dy = y - py
        dtheta = yaw_to_pi(yaw - pyaw)

        self.prev_odom_pose = (x, y, yaw)

        trans = math.hypot(dx, dy)
        direction = math.atan2(dy, dx)

        for p in self.particles:
            noisy_trans = trans + random.gauss(0.0, self.trans_noise_std)
            noisy_rot = dtheta + random.gauss(0.0, self.rot_noise_std)

            p.x += noisy_trans * math.cos(direction)
            p.y += noisy_trans * math.sin(direction)
            p.theta = yaw_to_pi(p.theta + noisy_rot)

        self.publish_particles()

    def publish_particles(self) -> None:
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        for p in self.particles:
            pose = Pose()
            pose.position.x = float(p.x)
            pose.position.y = float(p.y)
            pose.position.z = 0.0
            pose.orientation = yaw_to_quat(float(p.theta))
            msg.poses.append(pose)

        self.pub_particles.publish(msg)


def main():
    rclpy.init()
    node = MCLPrediction()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()