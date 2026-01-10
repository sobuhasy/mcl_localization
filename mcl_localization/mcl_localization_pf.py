import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def wrap_to_pi(angle: float) -> float:
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


class MCLLocalizationPF(Node):
    """
    Task A2 + A3:
      - Motion update from /robot_noisy (odometry)
      - Measurement update from /landmarks_observed using /landmarks_gt map
      - Gaussian sensor model
      - Normalize weights
    """

    def __init__(self):
        super().__init__("mcl_localization_pf")

        # ---------- Parameters ----------
        self.declare_parameter("num_particles", 200)
        self.declare_parameter("x_min", -10.0)
        self.declare_parameter("x_max", 10.0)
        self.declare_parameter("y_min", -10.0)
        self.declare_parameter("y_max", 10.0)
        self.declare_parameter("frame_id", "map")

        # Motion noise std devs
        self.declare_parameter("trans_noise_std", 0.05)  # meters
        self.declare_parameter("rot_noise_std", 0.02)  # radians

        # Measurement noise variance (matches your fake_robot param name)
        self.declare_parameter("measurement_noise_variance", 0.02)

        # Topics
        self.declare_parameter("odom_topic", "/robot_noisy")
        self.declare_parameter("map_landmarks_topic", "/landmarks_gt")
        self.declare_parameter("observed_landmarks_topic", "/landmarks_observed")

        # ---------- Read params ----------
        self.N = int(self.get_parameter("num_particles").value)
        self.x_min = float(self.get_parameter("x_min").value)
        self.x_max = float(self.get_parameter("x_max").value)
        self.y_min = float(self.get_parameter("y_min").value)
        self.y_max = float(self.get_parameter("y_max").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.trans_noise_std = float(self.get_parameter("trans_noise_std").value)
        self.rot_noise_std = float(self.get_parameter("rot_noise_std").value)
        self.meas_var = float(self.get_parameter("measurement_noise_variance").value)
        self.meas_sigma = math.sqrt(max(self.meas_var, 1e-12))

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.map_topic = str(self.get_parameter("map_landmarks_topic").value)
        self.obs_topic = str(self.get_parameter("observed_landmarks_topic").value)

        # ---------- State ----------
        self.particles: List[Particle] = self._init_uniform_particles(self.N)
        self.weights: List[float] = [1.0 / self.N] * self.N

        self.prev_odom_pose: Optional[Tuple[float, float, float]] = None

        # Map: id -> (x,y) in map frame
        self.landmark_map: Dict[int, Tuple[float, float]] = {}

        # ---------- ROS I/O ----------
        self.pub_particles = self.create_publisher(PoseArray, "particles", 10)

        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)
        self.sub_map = self.create_subscription(PointCloud2, self.map_topic, self.on_map_landmarks, 10)
        self.sub_obs = self.create_subscription(PointCloud2, self.obs_topic, self.on_observed_landmarks, 50)

        self.publish_particles()
        self.get_logger().info(
            f"MCLLocalizationPF started. N={self.N}, odom={self.odom_topic}, map={self.map_topic}, "
            f"obs={self.obs_topic}, meas_sigma={self.meas_sigma:.4f}"
        )

    def _init_uniform_particles(self, N: int) -> List[Particle]:
        out = []
        for _ in range(N):
            x = random.uniform(self.x_min, self.x_max)
            y = random.uniform(self.y_min, self.y_max)
            th = random.uniform(-math.pi, math.pi)
            out.append(Particle(x=x, y=y, theta=th))
        return out

    # ------------------ Task A2: Motion update ------------------

    def on_odom(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = wrap_to_pi(quat_to_yaw(msg.pose.pose.orientation))

        if self.prev_odom_pose is None:
            self.prev_odom_pose = (x, y, yaw)
            return

        px, py, pyaw = self.prev_odom_pose
        dx = x - px
        dy = y - py
        dtheta = wrap_to_pi(yaw - pyaw)
        self.prev_odom_pose = (x, y, yaw)

        trans = math.hypot(dx, dy)
        direction = math.atan2(dy, dx)

        for p in self.particles:
            noisy_trans = trans + random.gauss(0.0, self.trans_noise_std)
            noisy_rot = dtheta + random.gauss(0.0, self.rot_noise_std)

            p.x += noisy_trans * math.cos(direction)
            p.y += noisy_trans * math.sin(direction)
            p.theta = wrap_to_pi(p.theta + noisy_rot)

        self.publish_particles()

    # ------------------ Map landmarks subscriber ------------------

    def on_map_landmarks(self, msg: PointCloud2) -> None:
        # Expect fields: x(float32), y(float32), z(float32), id(int32)
        lm = {}
        for pt in point_cloud2.read_points(msg, field_names=("x", "y", "id"), skip_nans=True):
            x, y, lid = pt
            lm[int(lid)] = (float(x), float(y))

        if lm:
            self.landmark_map = lm
            self.get_logger().info(
                f"Loaded landmark map with {len(self.landmark_map)} landmarks "
                f"(from {msg.header.frame_id})."
            )

    # ------------------ Task A3: Measurement update ------------------

    def on_observed_landmarks(self, msg: PointCloud2) -> None:
        if not self.landmark_map:
            return

        observations = []
        for pt in point_cloud2.read_points(msg, field_names=("x", "y", "id"), skip_nans=True):
            xr, yr, lid = pt
            lid = int(lid)
            if lid in self.landmark_map:
                observations.append((float(xr), float(yr), lid))

        if not observations:
            return

        inv_2sig2 = 1.0 / (2.0 * self.meas_sigma * self.meas_sigma)

        logw = []
        for p in self.particles:
            c = math.cos(p.theta)
            s = math.sin(p.theta)

            ll = 0.0
            for (xr, yr, lid) in observations:
                xm = p.x + c * xr - s * yr
                ym = p.y + s * xr + c * yr

                xg, yg = self.landmark_map[lid]
                ex = xm - xg
                ey = ym - yg
                e2 = ex * ex + ey * ey

                ll += -e2 * inv_2sig2

            logw.append(ll)

        maxll = max(logw)
        w = [math.exp(lw - maxll) for lw in logw]
        s_w = sum(w)
        if s_w <= 0.0 or not math.isfinite(s_w):
            self.weights = [1.0 / self.N] * self.N
        else:
            self.weights = [wi / s_w for wi in w]

        best = max(self.weights)
        worst = min(self.weights)
        self.get_logger().info(f"Measurement update: best weight={best:.6f}, worst weight={worst:.6f}")

        neff = self.neff()
        if neff < 0.5 * self.N:
            self.get_logger().info(f"Resampling particles (neff={neff:.1f})")
            self.low_variance_resample()

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

    def neff(self) -> float:
        return 1.0 / sum(w * w for w in self.weights)
    
    def low_variance_resample(self) -> None:
        N = self.N
        new_particles: List[Particle] = []
        r = random.uniform(0, 1.0 / N)
        c = self.weights[0]
        i = 0

        for m in range(N):
            U = r + m / N
            while U > c and i < N - 1:
                i += 1
                c += self.weights[i]
            p = self.particles[i]
            # copy particle
            new_particles.append(Particle(x=p.x, y=p.y, theta=p.theta))

        self.particles = new_particles
        self.weights = [1.0 / N] * N


def main():
    rclpy.init()
    node = MCLLocalizationPF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()