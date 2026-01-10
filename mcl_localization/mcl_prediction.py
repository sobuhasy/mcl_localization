import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose, Quaternion

def yaw_to_pi(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q

def quat_to_yaw(q: Quaternion) -> float:
    # For planar yaw-only quaternions (where x=y=0), yaw = 2*atan2(z, w)
    return 2 * math.atan2(q.z, q.w)

@dataclass
class Particle:
    x: float
    y: float
    theta: float


class MCLPrediction(Node):
    """
    Task A2: Motion update.
    - Initialize particles uniformly (Task A1)
    - Subscribe to /robot_noisy (odometry)
    - Apply incremental odometry to each particle
    - Add Gaussian noise to translation and rotation
    - Normalize theta to [-pi, pi]
    - Publish PoseArray on /particles
    """

    def __init__(self):
        self.declare_parameter("num_particles", 100)
        self.declare_parameter("x_min", -10.0)
        self.declare_parameter("x_max", 10.0)
        self.declare_parameter("y_min", -10.0)
        self.declare_parameter("y_max", 10.0)
        self.declare_parameter("frame_id", "map")

        # Motion noise (std devs)
        # Translational noise (meters) applied to dx, dy increment
        self.declare_parameter("trans_noise_std", 0.05)
        # Rotational noise (radians) applied to dtheta increment
        self.declare_parameter("rot_noise_std", 0.02)

        # Odometry topic
        self.declare_parameter("odometry_topic", "/robot_noisy")

        self.N = int(self.get_parameter("num_particles").value)
        self.x_min = float(self.get_parameter("x_min").value)
        self.x_max = float(self.get_parameter("x_max").value)
        self.y_min = float(self.get_parameter("y_min").value)
        self.y_max = float(self.get_parameter("y_max").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.trans_noise_std = float(self.get_parameter("trans_noise_std").value)
        self.rot_noise_std = float(self.get_parameter("rot_noise_std").value)
        self.odom_topic = str(self.get_parameter("odometry_topic").value)

        # ---- State ----
        self.particles: List[Particle] = self._sample_uniform_particles(self.N)
        # self.weights: [1.0 / self.N] * self.N 

        self.prev_odom_pose: Optional[Tuple[float, float, float]] = None

        # ---- ROS I/O ----
        self.pub_particles = self.create_publisher(PoseArray, "particles", 10)
        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self.on_odom, 50
        )

        # Publish initial set immediately (so RViz shows something before first odom msg)
        self.publish_particles()

        self.get_logger().info(
            f"MCLPredicition started. N={self.N}, odom_topic={self.odom_topic}, "
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
        
    def on_odom(self, msg: Odometry):
        # Extract planar pose from odometry
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        yaw = wrap_to_pi(yaw)

        if self.prev_odom_pose is None:
            # First message: just store baseline
            self.prev_odom_pose = (x, y, yaw)
            self.publish_particles()
            return
        
        px, py, pyaw = self.prev_odom_pose

        # Incremental motion in the odom/map frame
        dx = x - px
        dy = y - py
        dtheta = yaw_to_pi(yaw - pyaw)

        self.prev_odom_pose = (x, y, yaw)

        # Apply motion to each particle using standard odom-based model:
        # - Interpret (dx, dy, dtheta) as translation in the global frame, but apply in particle frame.
        # Equivalent: convert global translation into particle-local, then forward in particle heading.
        # 
        # Alternative simpler approach: apply dx, dy directly to particle position (global).
        # For your fake_robot topics (already in map frame), that simpler approach is fine.
        # We'll do the robust one: compute translation magnitude and direction in global,
        # then apply it relative to particle theta.

        trans = math.hypot(dx, dy)
        direction = math.atan2(dy, dx)

        for i, p in enumerate(self.particles):
            # Sample noise
            noisy_trans = trans + random.gauss(0.0, self.trans_noise_std)
            noisy_rot = dtheta + random.gauss(0.0, self.rot_noise_std)

            # Apply translation along the *measured* motion direction,
            # but expressed relative to particle pose:
            # global heading of motion = direction
            # so displacement components are:
            p.x += noisy_trans * math.cos(direction)
            p.y += noisy_trans * math.sin(direction)

            # Update orientation
            p.theta = yaw_to_pi(p.theta + noisy_rot)

        # Publish updated particle set
        self.publish_particles()

    def publish_particles(self) -> None:
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        for p in self.particles:
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            pose.position.z = 0.0
            pose.orientation = yaw_to_quat(p.theta)
            msg.poses.append(pose)

        self.pub_particles.publish(msg)

def main():
    rclpy.init()
    node = MCLPrediction()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()