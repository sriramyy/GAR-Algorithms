#!/usr/bin/env python3
import math
from enum import Enum, auto
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

# ------------------------------------------------------------
# FSM
# ------------------------------------------------------------
class OvertakeState(Enum):
    FOLLOW = auto()
    PREPARE = auto()
    OVERTAKE = auto()
    RETURN = auto()

# ------------------------------------------------------------
# Target representation
# ------------------------------------------------------------
class TargetVehicle:
    def __init__(self, pose: Pose, vel: float):
        self.pose = pose
        self.vel = vel

# ------------------------------------------------------------
# Node
# ------------------------------------------------------------
class OvertakeFollowGap(Node):
    def __init__(self):
        super().__init__("overtake_follow_gap")

        # ---------- topic names ----------
        self.scan_topic = "/scan"
        self.odom_topic = "/odom"
        self.opps_topic = "/opponents"
        self.drive_topic = "/drive"

        # ---------- ROS2 I/O ----------
        qos10 = QoSProfile(depth=10)
        qos20 = QoSProfile(depth=20)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.lidar_callback, qos10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos20)
        self.opps_sub = self.create_subscription(PoseArray, self.opps_topic, self.opponents_callback, qos10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, qos10)

        # ---------- FOLLOW-gap knobs ----------
        self.BUBBLE_RADIUS = 100
        self.PREPROCESS_CONV_SIZE = 3
        self.BEST_POINT_CONV_SIZE = 120
        self.MAX_LIDAR_DIST = 7.0
        self.MAX_STEER_ABS = np.deg2rad(40.0)

        # speeds
        self.STRAIGHT_SPEED = 3.0
        self.CORNER_SPEED  = 2.0
        self.SPEED_MAX     = 4.5

        # handling
        self.CENTER_BIAS_ALPHA = 0.35
        self.EDGE_GUARD_DEG   = 12.0
        self.SIDE_REPULSION_GAIN = 0.28
        self.TTC_HARD_BRAKE   = 0.55
        self.TTC_SOFT_BRAKE   = 0.9
        self.FWD_WEDGE_DEG    = 8.0
        self.STEER_SMOOTH_ALPHA = 0.5
        self.STEER_RATE_LIMIT   = np.deg2rad(8.0)

        # ---------- OVERTAKING knobs ----------
        self.FOLLOW_TIME_GAP = 0.8
        self.MIN_SPEED_ADV   = 0.3
        self.MIN_CLEAR_DIST  = 3.0
        self.PASS_SIDE       = "left"
        self.PASS_BIAS_DEG   = 18.0
        self.RETURN_LATENCY  = 0.8
        self.PREPARE_TIMEOUT = 2.0

        # ---------- prediction knobs ----------
        self.PREDICTION_TIME = 1.0
        self.OPEN_GAP_THRESHOLD = 4.5
        self.PREDICTION_ENABLED = True

        # ---------- overtaking boost ----------
        self.OVERTAKE_SPEED_BOOST = 0.4

        # ---------- NEW SAFETY features ----------
        self.MIN_RETURN_GAP = 2.5
        self.MIN_LATERAL_CLEAR = 1.0
        self.PASS_FUTURE_HORIZON = 0.7

        self.CAR_LENGTH = 0.4       # approximate F1TENTH car length in meters
        self.FRONT_MARGIN = 0.23    # extra buffer in front of car
        self.PASS_TIME_EST = 1.0        # how far ahead in time we check front gap when returning

        # target persistence
        self.target_memory = None
        self.target_memory_ts = 0.0
        self.TARGET_MEMORY_TIME = 0.8

        self.COLLISION_LONG_THRESH = 3.5 # not used yet
        self.COLLISION_LAT_THRESH = 1.0

        self.EXTRA_OVERTAKE_BUBBLE = 80

        # opponent-side repulsion
        self.OPP_REPULSION_GAIN = 0.35

        # ---------- runtime state ----------
        self.state = OvertakeState.FOLLOW
        self.state_ts = self.now_sec()
        self.ego_pose: Optional[Pose] = None
        self.ego_speed: float = 0.0
        self.opponents: List[TargetVehicle] = []
        self.radians_per_elem: Optional[float] = None
        self.proc_latest: Optional[np.ndarray] = None
        self.prev_steer: float = 0.0

        self.get_logger().info("OvertakeFollowGap (front-safe) started.")


    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def yaw_of(self, pose: Pose) -> float:
        q = pose.orientation
        return math.atan2(
            2.0*(q.w*q.z + q.x*q.y),
            1.0 - 2.0*(q.z*q.z + q.y*q.y)
        )

    # predict how far IN FRONT of the target our ego car will be after dt seconds
    # positive = ego ahead, negative = ego behind
    def predict_front_gap(self, target: TargetVehicle, dt: float) -> float:
        if self.ego_pose is None:
            return 0.0
        # current forward separation (target -> ego)
        s_now = self.longitudinal_gap(target.pose, self.ego_pose)
        # relative speed along lane (ego - target)
        rel_v = self.ego_speed - target.vel
        return s_now + rel_v * dt

    def odom_callback(self, msg: Odometry):
        self.ego_pose = msg.pose.pose
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.ego_speed = float(np.hypot(vx, vy))

    def opponents_callback(self, msg: PoseArray):
        # placeholder constant velocity assumption (slow-ish)
        self.opponents = [TargetVehicle(p, 2.5) for p in msg.poses]

    def lidar_callback(self, scan: LaserScan):
        if self.ego_pose is None:
            return

        ranges_full = np.array(scan.ranges, dtype=np.float32)
        proc = self.preprocess_lidar(ranges_full)
        self.proc_latest = proc.copy()

        # ---------- target selection with memory ----------
        target = self.select_front_target_stable()

        # ---------- prediction (target future pose) ----------
        predicted = None
        if target and self.PREDICTION_ENABLED:
            predicted = self.predict_future(target)

        # ---------- FSM ----------
        self.step_fsm(target, predicted)

        # ---------- gap follower core ----------
        if proc.size == 0:
            return

        closest_idx = int(np.argmin(proc))

        # expand safety bubble during overtake
        radius = self.BUBBLE_RADIUS
        if self.state == OvertakeState.OVERTAKE:
            radius += self.EXTRA_OVERTAKE_BUBBLE

        proc = self.mask_bubble(proc, closest_idx, radius)
        gap_start, gap_end = self.find_max_gap(proc)
        best_idx = self.find_best_point(gap_start, gap_end, proc)

        # edge guard
        best_idx = self.apply_edge_guard(best_idx, proc.size)

        # center bias
        best_idx = self.apply_center_bias(best_idx, proc.size, self.CENTER_BIAS_ALPHA)

        # overtaking pass bias
        if self.state in (OvertakeState.PREPARE, OvertakeState.OVERTAKE):
            bias = self.index_bias(proc.size, np.deg2rad(self.PASS_BIAS_DEG))
            best_idx = int(np.clip(best_idx + bias, 0, proc.size - 1))

        # opponent repulsion
        best_idx = self.apply_opponent_repulsion(best_idx, proc.size, target)

        # side repulsion from walls
        repel = self.side_repulsion_shift(proc)
        best_idx = int(np.clip(best_idx + repel, 0, proc.size - 1))

        # ---------- steering ----------
        steer_raw = self.index_to_steer(best_idx, proc.size)
        steer_cmd = self.smooth_and_limit_steer(steer_raw)

        # ---------- speed ----------
        speed_cmd = self.speed_policy(steer_cmd)

        # ---------- publish ----------
        self.publish_drive(speed_cmd, steer_cmd)

    # ------------------------------------------------------------
    # Target selection with *stability*
    # ------------------------------------------------------------
    def select_front_target_stable(self) -> Optional[TargetVehicle]:
        now = self.now_sec()
        raw = self.select_front_target()

        if raw is None:
            if self.target_memory and now - self.target_memory_ts < self.TARGET_MEMORY_TIME:
                return self.target_memory
            return None

        self.target_memory = raw
        self.target_memory_ts = now
        return raw

    def select_front_target(self) -> Optional[TargetVehicle]:
        if self.ego_pose is None or not self.opponents:
            return None

        best = None
        best_s = float("inf")
        yaw = self.yaw_of(self.ego_pose)

        for o in self.opponents:
            dx = o.pose.position.x - self.ego_pose.position.x
            dy = o.pose.position.y - self.ego_pose.position.y
            s = math.cos(yaw)*dx + math.sin(yaw)*dy   # forward
            l = -math.sin(yaw)*dx + math.cos(yaw)*dy  # lateral
            if s > 0 and s < best_s and abs(l) < 3.0:
                best = o
                best_s = s
        return best

    # ------------------------------------------------------------
    # Longitudinal & lateral geometry
    # ------------------------------------------------------------
    def longitudinal_gap(self, ego: Pose, other: Pose) -> float:
        yaw = self.yaw_of(ego)
        dx = other.position.x - ego.position.x
        dy = other.position.y - ego.position.y
        return math.cos(yaw)*dx + math.sin(yaw)*dy

    def lateral_gap(self, ego: Pose, other: Pose) -> float:
        yaw = self.yaw_of(ego)
        dx = other.position.x - ego.position.x
        dy = other.position.y - ego.position.y
        return -math.sin(yaw)*dx + math.cos(yaw)*dy

    def passed_target(self, target: TargetVehicle) -> bool:
        # target -> ego forward distance; require at least half car length + margin
        if self.ego_pose is None:
            return False
        s = self.longitudinal_gap(target.pose, self.ego_pose)
        return s > (self.CAR_LENGTH / 2.0 + self.FRONT_MARGIN)

    # ------------------------------------------------------------
    # Prediction (target pose only – kept for compatibility)
    # ------------------------------------------------------------
    def predict_future(self, target: TargetVehicle) -> TargetVehicle:
        yaw_ego = self.yaw_of(self.ego_pose)
        dx = target.vel * self.PREDICTION_TIME * math.cos(yaw_ego)
        dy = target.vel * self.PREDICTION_TIME * math.sin(yaw_ego)
        fp = Pose()
        fp.position.x = target.pose.position.x + dx
        fp.position.y = target.pose.position.y + dy
        fp.orientation = target.pose.orientation
        return TargetVehicle(fp, target.vel)

    # ------------------------------------------------------------
    # FSM with front-gap-based safety when returning
    # ------------------------------------------------------------
    def step_fsm(self, target, predicted):
        now = self.now_sec()
        t_state = now - self.state_ts

        if target is None:
            if self.state != OvertakeState.FOLLOW:
                self.set_state(OvertakeState.RETURN)
            return

        dist = self.longitudinal_gap(self.ego_pose, target.pose)
        lat  = self.lateral_gap(self.ego_pose, target.pose)
        rel_v = self.ego_speed - target.vel
        desired_gap = self.FOLLOW_TIME_GAP * max(self.ego_speed, 0.1)

        # ---------- FOLLOW ----------
        if self.state == OvertakeState.FOLLOW:
            if dist < desired_gap and rel_v < self.MIN_SPEED_ADV:
                self.set_state(OvertakeState.PREPARE)

        # ---------- PREPARE ----------
        elif self.state == OvertakeState.PREPARE:
            if predicted:
                # keep your original prediction-based gap check
                future_gap = self.longitudinal_gap(self.ego_pose, predicted.pose)
                if (future_gap >= self.OPEN_GAP_THRESHOLD and
                    self.clear_on_pass_side()):
                    self.set_state(OvertakeState.OVERTAKE)
            if t_state > self.PREPARE_TIMEOUT:
                self.set_state(OvertakeState.FOLLOW)

        # ---------- OVERTAKE ----------
        elif self.state == OvertakeState.OVERTAKE:
            # predict how far ahead we’ll be after a short horizon
            gap_future = self.predict_front_gap(target, self.PASS_TIME_EST)
            enough_front_gap = gap_future >= (self.CAR_LENGTH + self.FRONT_MARGIN)
            if (self.passed_target(target) and
                enough_front_gap and
                self.forward_clearance() >= self.MIN_CLEAR_DIST and
                abs(lat) >= self.MIN_LATERAL_CLEAR):
                self.set_state(OvertakeState.RETURN)

        # ---------- RETURN ----------
        elif self.state == OvertakeState.RETURN:
            if t_state > self.RETURN_LATENCY:
                self.set_state(OvertakeState.FOLLOW)

    def set_state(self, s: OvertakeState):
        if self.state != s:
            self.state = s
            self.state_ts = self.now_sec()
            self.get_logger().info(f"STATE -> {self.state.name}")

    # ------------------------------------------------------------
    # Opponent-side repulsion (safety feature)
    # ------------------------------------------------------------
    def apply_opponent_repulsion(self, idx: int, length: int, target: Optional[TargetVehicle]) -> int:
        if target is None or self.radians_per_elem is None:
            return idx
        lat = self.lateral_gap(self.ego_pose, target.pose)
        if lat > 0:
            repel = -self.OPP_REPULSION_GAIN / (abs(lat) + 0.2)
        else:
            repel = +self.OPP_REPULSION_GAIN / (abs(lat) + 0.2)
        shift = int(round(repel / (self.radians_per_elem or 1e-6)))
        return int(np.clip(idx + shift, 0, length - 1))

    # ------------------------------------------------------------
    # LIDAR preprocessing
    # ------------------------------------------------------------
    def preprocess_lidar(self, ranges: np.ndarray) -> np.ndarray:
        n = len(ranges)
        self.radians_per_elem = (2.0 * np.pi) / n if n > 0 else None

        if n > 270:
            proc = ranges[135:-135].copy()
        else:
            proc = ranges.copy()

        if self.PREPROCESS_CONV_SIZE > 1:
            k = np.ones(self.PREPROCESS_CONV_SIZE, dtype=np.float32) / float(self.PREPROCESS_CONV_SIZE)
            proc = np.convolve(proc, k, mode="same")

        np.clip(proc, 0.0, self.MAX_LIDAR_DIST, out=proc)
        return proc

    def mask_bubble(self, arr: np.ndarray, center: int, radius: int) -> np.ndarray:
        a = arr.copy()
        if a.size == 0:
            return a
        lo = max(0, center - radius)
        hi = min(a.size, center + radius + 1)
        a[lo:hi] = 0.0
        return a

    # ------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------
    def find_max_gap(self, arr: np.ndarray):
        if arr.size == 0:
            return 0, 0
        masked = np.ma.masked_where(arr == 0.0, arr)
        spans = np.ma.notmasked_contiguous(masked)
        if not spans:
            return 0, arr.size
        best = max(spans, key=lambda sl: (sl.stop - sl.start))
        return best.start, best.stop

    def find_best_point(self, start: int, stop: int, arr: np.ndarray) -> int:
        if arr.size == 0:
            return 0
        if stop <= start + 1:
            return start
        seg = arr[start:stop]
        if self.BEST_POINT_CONV_SIZE > 1:
            k = np.ones(self.BEST_POINT_CONV_SIZE, dtype=np.float32) / float(self.BEST_POINT_CONV_SIZE)
            seg = np.convolve(seg, k, mode="same")
        return int(np.argmax(seg)) + start

    def apply_edge_guard(self, idx: int, length: int) -> int:
        if self.radians_per_elem is None or length == 0:
            return 0
        guard = int(round(np.deg2rad(self.EDGE_GUARD_DEG) / self.radians_per_elem))
        lo = guard
        hi = length - guard - 1
        return int(np.clip(idx, lo, hi))

    def apply_center_bias(self, idx: int, length: int, alpha: float) -> int:
        center = (length - 1) / 2.0
        biased = (1 - alpha) * idx + alpha * center
        return int(np.clip(round(biased), 0, length - 1))

    def side_repulsion_shift(self, proc: np.ndarray) -> int:
        if proc.size == 0 or self.radians_per_elem is None:
            return 0

        L = proc.size
        band = max(6, int(0.05 * L))
        left_avg  = float(np.mean(proc[:band]))
        right_avg = float(np.mean(proc[-band:]))
        diff = right_avg - left_avg  # >0 → closer on left → steer right

        per = self.radians_per_elem
        max_shift = int(round(self.SIDE_REPULSION_GAIN / per))
        shift = int(np.clip(np.sign(diff) * min(abs(diff), 1.0) * max_shift,
                            -max_shift, max_shift))
        return shift

    # ------------------------------------------------------------
    # Steering & speed
    # ------------------------------------------------------------
    def index_to_steer(self, idx: int, length: int) -> float:
        if self.radians_per_elem is None:
            return 0.0
        angle = (idx - (length / 2.0)) * self.radians_per_elem
        steer = angle / 2.0
        return float(np.clip(steer, -self.MAX_STEER_ABS, self.MAX_STEER_ABS))

    def smooth_and_limit_steer(self, steer: float) -> float:
        s = (1.0 - self.STEER_SMOOTH_ALPHA) * steer + \
            self.STEER_SMOOTH_ALPHA * self.prev_steer
        delta = np.clip(s - self.prev_steer,
                        -self.STEER_RATE_LIMIT, self.STEER_RATE_LIMIT)
        s_lim = self.prev_steer + float(delta)
        self.prev_steer = s_lim
        return s_lim

    def forward_clearance(self) -> float:
        if self.proc_latest is None or self.radians_per_elem is None:
            return self.MAX_LIDAR_DIST
        a = self.proc_latest
        L = a.size
        center = L // 2
        half = int(round(np.deg2rad(self.FWD_WEDGE_DEG) / self.radians_per_elem))
        lo = max(0, center - half)
        hi = min(L, center + half + 1)
        return float(np.min(a[lo:hi]))

    def speed_policy(self, steer: float) -> float:
        base = self.CORNER_SPEED if abs(steer) > np.deg2rad(10) else self.STRAIGHT_SPEED
        base = min(base, self.SPEED_MAX)

        # forward TTC braking
        fwd = self.forward_clearance()
        v = max(self.ego_speed, 0.05)
        ttc = fwd / v
        if ttc < self.TTC_HARD_BRAKE:
            base = 0.0
        elif ttc < self.TTC_SOFT_BRAKE:
            scale = (ttc - self.TTC_HARD_BRAKE) / (self.TTC_SOFT_BRAKE - self.TTC_HARD_BRAKE)
            base = scale * base

        if abs(steer) < np.deg2rad(6.0) and fwd > 3.0:
            base = min(self.SPEED_MAX, base + 0.5)

        if self.state == OvertakeState.OVERTAKE:
            base = min(self.SPEED_MAX, base + self.OVERTAKE_SPEED_BOOST)

        return base

    # ------------------------------------------------------------
    # Pass-side clearance check
    # ------------------------------------------------------------
    def clear_on_pass_side(self) -> bool:
        if self.proc_latest is None or self.radians_per_elem is None:
            return False

        a = self.proc_latest
        L = a.size
        center = L // 2
        per = self.radians_per_elem

        wedge_center = np.deg2rad(15.0)
        wedge_half   = np.deg2rad(8.0)

        off_idx  = int(round(wedge_center / per))
        half_idx = int(round(wedge_half   / per))

        if self.PASS_SIDE == "left":
            c = center - off_idx
        else:
            c = center + off_idx

        lo = max(0, c - half_idx)
        hi = min(L, c + half_idx + 1)
        if hi <= lo:
            return False

        min_val = float(np.min(a[lo:hi]))
        return min_val >= self.MIN_CLEAR_DIST

    # ------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------
    def publish_drive(self, speed: float, steer: float):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(np.clip(steer, -self.MAX_STEER_ABS, self.MAX_STEER_ABS))
        msg.drive.speed = float(max(0.0, speed))
        self.drive_pub.publish(msg)

# ------------------------------------------------------------
# main()
# ------------------------------------------------------------
def main():
    rclpy.init()
    node = OvertakeFollowGap()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()