import csv
import math
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_yaw(z: float, w: float) -> float:
    """Convert quaternion (z, w) to yaw angle."""
    return 2 * math.atan2(z, w)

@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
    t: float

class PFEvaluator(Node):
    def __init__(self):
        super().__init__('pf_evaluator')

        self.declare_parameter('ground_truth_topic', '/ground_truth_pose')
        self.declare_parameter('estimated_pose_topic', '/estimated_pose')
        self.declare_parameter('output_csv', 'pf_evaluation.csv')
        self.declare_parameter('rmse_only_xy', True)
        
        # convergence settings
        self.declare_parameter('conv_xy_threshold', 0.25)
        self.declare_parameter('conv_hold_time', 2.0)

        self.gt_topic = self.get_parameter('ground_truth_topic').value
        self.est_topic = self.get_parameter('estimated_pose_topic').value
        self.output_csv = self.get_parameter('output_csv').value
        self.rmse_only_xy = bool(self.get_parameter('rmse_only_xy').value)

        self.conv_xy_threshold = float(self.get_parameter('conv_xy_threshold').value)
        self.conv_hold_time = float(self.get_parameter('conv_hold_time').value)

        self.gt: Optional[Pose2D] = None
        self.est: Optional[Pose2D] = None

        self.sum_squared_error_xy = 0.0
        self.sum_squared_error_yaw = 0.0
        self.num_samples = 0

        self.conv_start: Optional[float] = None
        self.conv_time: Optional[float] = None

        # open CSV file
        self.csv_file = open(self.output_csv, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time', 'error_x', 'error_y', 'error_yaw', 'rmse_xy', 'rmse_yaw'])

        # subscribers
        self.create_subscription(PoseStamped, self.gt_topic, self.gt_callback, 50)
        self.create_subscription(PoseStamped, self.est_topic, self.est_callback, 50)

        self.get_logger().info(f'Logging to {self.output_csv}(gt: {self.gt_topic}, est: {self.est_topic})')

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9
    
    def gt_callback(self, msg: Odometry):
        t = self.now_s()
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        yaw = wrap_to_pi(quat_to_yaw(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        self.gt = Pose2D(x, y, yaw, t)
        self.evaluate()

    def est_callback(self, msg: Odometry):
        t = self.now_s()
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        yaw = wrap_to_pi(quat_to_yaw(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        self.est = Pose2D(x, y, yaw, t)
        self.evaluate()

    def try_update(self):
        if self.gt is None or self.est is None:
            return
        
        # use latest pair
        t = min(self.gt.t, self.est.t)

        dx = self.est.x - self.gt.x
        dy = self.est.y - self.gt.y
        e_xy = math.sqrt(dx * dx + dy * dy)
        e_yaw = wrap_to_pi(self.est.yaw - self.gt.yaw)

        # accumulate RMSE
        self.sum_squared_error_xy += e_xy * e_xy
        self.sum_squared_error_yaw += e_yaw * e_yaw
        self.num_samples += 1

        # convergence logic: hold below threshold for conv_hold_time
        if e_xy < self.conv_xy_threshold:
            if self.conv_start is None:
                self.conv_start = t
            elif t - self.conv_start >= self.conv_hold_time:
                if self.conv_time is None:
                    self.conv_time = t - self.conv_start
        else:
            self.conv_start = None

        self.writer.writerow([t, self.est.x - self.gt.x, self.est.y - self.gt.y, e_yaw,
                              math.sqrt(self.sum_squared_error_xy / self.num_samples),
                              math.sqrt(self.sum_squared_error_yaw / self.num_samples)])

    def destroy_node(self):
        if self.n > 0:
            rmse_xy = math.sqrt(self.sum_squared_error_xy / self.num_samples)
            rmse_yaw = math.sqrt(self.sum_squared_error_yaw / self.num_samples)

            if self.conv_time is None:
                conv_str = "not converged"
            else:
                conv_str = f"{self.conv_time:.2f}s converged"

            if self.rmse_only_xy:
                self.get_logger().info(f"RMSE_xy={rmse_xy:.4f} m, convergence={conv_str}, samples={self.n}")
            else:
                self.get_logger().info(f"RMSE_xy={rmse_xy:.4f} m, RMSE_yaw={rmse_yaw:.4f} rad, convergence={conv_str}, samples={self.n}")

        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass
        super().destroy_node()

    def main():
        rclpy.init()
        node = PFEvaluator()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()