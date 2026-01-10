import math
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Quaternion


def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class ParticleInitializer(Node):
    def __init__(self):
        super().__init__("particle_initializer")

        # ---- Parameters (map bounds) ----
        self.declare_parameter("num_particles", 100)
        self.declare_parameter("x_min", -10.0)
        self.declare_parameter("x_max", 10.0)
        self.declare_parameter("y_min", -10.0)
        self.declare_parameter("y_max", 10.0)
        self.declare_parameter("theta_min", -math.pi)
        self.declare_parameter("theta_max", math.pi)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate_hz", 1.0)

        self.N = int(self.get_parameter("num_particles").value)
        self.x_min = float(self.get_parameter("x_min").value)
        self.x_max = float(self.get_parameter("x_max").value)
        self.y_min = float(self.get_parameter("y_min").value)
        self.y_max = float(self.get_parameter("y_max").value)
        self.t_min = float(self.get_parameter("theta_min").value)
        self.t_max = float(self.get_parameter("theta_max").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.publisher = self.create_publisher(PoseArray, "particles", 10)

        # ---- Create particles once (A1) ----
        self.particles = self._sample_uniform_particles(self.N)

        # Equal weights (store internally for later MCL steps)
        self.weights = [1.0 / self.N] * self.N

        rate = float(self.get_parameter("publish_rate_hz").value)
        period = 1.0 / max(rate, 1e-6)
        self.timer = self.create_timer(period, self.publish_particles)

        self.get_logger().info(
            f"Initialized {self.N} particles uniformly: "
            f"x[{self.x_min},{self.x_max}], y[{self.y_min},{self.y_max}], theta[{self.t_min},{self.t_max}]"
        )

    def _sample_uniform_particles(self, N: int):
        particles = []
        for _ in range(N):
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            th = random.uniform(self.t_min, self.t_max)
            particles.append((x, y, th))
        return particles

    def publish_particles(self):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        for (x, y, th) in self.particles:
            p = Pose()
            p.position.x = x
            p.position.y = y
            p.position.z = 0.0
            p.orientation = yaw_to_quat(th)
            msg.poses.append(p)

        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = ParticleInitializer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
